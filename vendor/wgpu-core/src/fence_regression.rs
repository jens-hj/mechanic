//! Dependency-local tests: real core acquisition/submission with a controllable HAL surface.
use super::*;
use crate::lock::{rank, Mutex};
use alloc::{boxed::Box, vec};
use std::{
    any::Any,
    sync::{mpsc, Barrier},
    thread,
    time::Duration,
};

struct PausedSurface {
    entered: mpsc::Sender<()>,
    resume: std::sync::Mutex<mpsc::Receiver<()>>,
    context: hal::noop::Context,
}
impl hal::DynResource for PausedSurface {
    fn as_any(&self) -> &dyn Any {
        &self.context
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.context
    }
}
impl hal::DynSurface for PausedSurface {
    unsafe fn configure(
        &self,
        _: &dyn hal::DynDevice,
        _: &hal::SurfaceConfiguration,
    ) -> Result<(), hal::SurfaceError> {
        Ok(())
    }
    unsafe fn unconfigure(&self, _: &dyn hal::DynDevice) {}
    unsafe fn acquire_texture(
        &self,
        _: Option<Duration>,
        _: &dyn hal::DynFence,
    ) -> Result<hal::DynAcquiredSurfaceTexture, hal::SurfaceError> {
        self.entered.send(()).unwrap();
        // Watchdog also releases acquisition if the controller panics or forgets cleanup.
        let _ = self
            .resume
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(5));
        Ok(hal::DynAcquiredSurfaceTexture {
            texture: Box::new(hal::noop::Resource),
            suboptimal: false,
        })
    }
    unsafe fn discard_texture(&self, _: Box<dyn hal::DynSurfaceTexture>) {}
}

#[test]
fn acquisition_does_not_block_unrelated_submission() {
    // A lost lock must fail the test process rather than hang CI indefinitely.
    let (watchdog_done, watchdog_rx) = mpsc::channel::<()>();
    let watchdog = thread::spawn(move || {
        if matches!(
            watchdog_rx.recv_timeout(Duration::from_secs(20)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ) {
            std::process::abort();
        }
    });
    let global = Global::new(
        "fence regression",
        wgt::InstanceDescriptor {
            backends: wgt::Backends::NOOP,
            backend_options: wgt::BackendOptions {
                noop: wgt::NoopBackendOptions { enable: true },
                ..Default::default()
            },
            ..wgt::InstanceDescriptor::new_without_display_handle()
        },
        None,
    );
    let adapter = global
        .request_adapter(&Default::default(), wgt::Backends::NOOP, None)
        .unwrap();
    let (device_id, queue_id) = global
        .adapter_request_device(adapter, &Default::default(), None, None)
        .unwrap();
    let device = global.hub.devices.get(device_id);
    let queue = global.hub.queues.get(queue_id);
    let (entered_tx, entered_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let hal_surface: Box<dyn hal::DynSurface> = Box::new(PausedSurface {
        entered: entered_tx,
        resume: std::sync::Mutex::new(resume_rx),
        context: hal::noop::Context,
    });
    let surface = Arc::new(Surface {
        presentation: Mutex::new(
            rank::SURFACE_PRESENTATION,
            Some(Presentation {
                device: device.clone(),
                config: wgt::SurfaceConfiguration {
                    usage: wgt::TextureUsages::RENDER_ATTACHMENT,
                    format: wgt::TextureFormat::Rgba8Unorm,
                    width: 1,
                    height: 1,
                    present_mode: wgt::PresentMode::Fifo,
                    desired_maximum_frame_latency: 2,
                    alpha_mode: wgt::CompositeAlphaMode::Opaque,
                    view_formats: vec![],
                },
                acquired_texture: None,
            }),
        ),
        surface_per_backend: [(wgt::Backend::Noop, hal_surface)].into_iter().collect(),
    });
    let acquiring = surface.clone();
    let acquisition = thread::spawn(move || acquiring.get_current_texture().unwrap());
    let arrived = entered_rx.recv_timeout(Duration::from_secs(5));
    let (submitted_tx, submitted_rx) = mpsc::channel();
    let submitting = queue.clone();
    let submission = thread::spawn(move || submitted_tx.send(submitting.submit(&[])).unwrap());
    let progressed = submitted_rx.recv_timeout(Duration::from_secs(1));
    // Always unblock acquisition before asserting, including on the unpatched baseline.
    let _ = resume_tx.send(());
    let output = acquisition.join().unwrap();
    submission.join().unwrap();
    assert!(arrived.is_ok(), "acquisition did not reach HAL boundary");
    assert!(
        progressed.is_ok(),
        "submission stalled behind HAL acquisition"
    );
    assert!(progressed.unwrap().is_ok());
    drop(output);
    surface.present().unwrap();

    // Concurrent submit/poll/present, synchronized start; callbacks re-enter submit
    // to prove they run outside core locks. Each producer must see increasing indices.
    let callbacks_run = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(3));
    thread::scope(|scope| {
        for _ in 0..2 {
            let queue = queue.clone();
            let barrier = barrier.clone();
            let callbacks_run = callbacks_run.clone();
            scope.spawn(move || {
                barrier.wait();
                let mut previous = 0;
                for _ in 0..50 {
                    let index = queue.submit(&[]).unwrap();
                    assert!(index > previous);
                    previous = index;
                    let callback_queue = queue.clone();
                    let callbacks_run = callbacks_run.clone();
                    queue.on_submitted_work_done(Box::new(move || {
                        callback_queue.submit(&[]).unwrap();
                        callbacks_run.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }));
                }
            });
        }
        barrier.wait();
        for _ in 0..50 {
            let _ = resume_tx.send(());
            drop(surface.get_current_texture().unwrap());
            surface.present().unwrap();
            device.poll(wgt::PollType::Poll).unwrap();
        }
    });
    device.poll(wgt::PollType::wait_indefinitely()).unwrap();
    assert_eq!(
        callbacks_run.load(std::sync::atomic::Ordering::Relaxed),
        100
    );
    drop(watchdog_done);
    watchdog.join().unwrap();
}

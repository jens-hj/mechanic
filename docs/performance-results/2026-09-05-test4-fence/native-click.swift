import CoreGraphics
import Foundation
let x = Double(CommandLine.arguments[1])!
let y = Double(CommandLine.arguments[2])!
let p = CGPoint(x:x,y:y)
CGEvent(mouseEventSource:nil,mouseType:.mouseMoved,mouseCursorPosition:p,mouseButton:.left)?.post(tap:.cghidEventTap)
Thread.sleep(forTimeInterval:0.15)
CGEvent(mouseEventSource:nil,mouseType:.leftMouseDown,mouseCursorPosition:p,mouseButton:.left)?.post(tap:.cghidEventTap)
Thread.sleep(forTimeInterval:0.1)
CGEvent(mouseEventSource:nil,mouseType:.leftMouseUp,mouseCursorPosition:p,mouseButton:.left)?.post(tap:.cghidEventTap)

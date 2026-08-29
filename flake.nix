{
  description = "Mechanic construction and simulation sandbox";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      rust-overlay,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };
        toolchainSpec = (builtins.fromTOML (builtins.readFile ./rust-toolchain.toml)).toolchain;
        rustToolchain = pkgs.rust-bin.stable.${toolchainSpec.channel}.default.override {
          extensions = toolchainSpec.components ++ [
            "rust-analyzer"
            "rust-src"
          ];
        };
        runtimeLibs = pkgs.lib.optionals pkgs.stdenv.isLinux (
          with pkgs;
          [
            vulkan-loader
            libGL
            wayland
            libxkbcommon
            libx11
            libxcursor
            libxi
            libxrandr
          ]
        );
        linuxEnvironment = pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;
        };
        developmentTools = [
          rustToolchain
          pkgs.git
          pkgs.openssh
          pkgs.pkg-config
        ];
        mechanic = pkgs.writeShellApplication {
          name = "mechanic";
          runtimeInputs = developmentTools;
          text = ''
            export CARGO_TARGET_DIR="''${CARGO_TARGET_DIR:-''${XDG_CACHE_HOME:-$HOME/.cache}/mechanic/target}"
            ${pkgs.lib.optionalString pkgs.stdenv.isLinux ''
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            ''}
            cd ${self}/crates/mechanic-app
            exec cargo run --manifest-path ${self}/Cargo.toml -p mechanic-app -- "$@"
          '';
        };
      in
      {
        devShells.default = pkgs.mkShell (
          {
            packages = developmentTools;
            buildInputs = runtimeLibs;
            RUST_BACKTRACE = "1";
          }
          // linuxEnvironment
        );

        apps.default = {
          type = "app";
          program = "${mechanic}/bin/mechanic";
          meta.description = "Build and run the Mechanic prototype";
        };

        formatter = pkgs.nixfmt;
      }
    );
}

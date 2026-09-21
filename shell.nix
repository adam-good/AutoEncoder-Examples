{ pkgs ? import <nixpkgs> {}} :
pkgs.mkShellNoCC {
    packages = with pkgs; [
        julia-bin
    ];

    shellHook = ''
        export JULIA_DEPOT_PATH=./juliapkgs
    '';
}

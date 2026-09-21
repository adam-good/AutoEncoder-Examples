{ pkgs ? import <nixpkgs> {}} :
pkgs.mkShellNoCC {
    packages = with pkgs; [
        python3
        julia-bin
    ];

    shellHook = ''
        export JULIA_DEPOT_PATH=./juliapkgs
    '';
}

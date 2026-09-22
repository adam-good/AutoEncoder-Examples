{ pkgs ? import <nixpkgs> {}} :
pkgs.mkShellNoCC {
    packages = with pkgs; [
        (pkgs.buildFHSEnv {
            name = "julia-fhs";
            targetPkgs = pkgs: with pkgs; [
                julia-bin
                qt5.qtbase
                fontconfig
                python3
            ];
         })
#        python3
#        julia-bin
#        qt5.qtbase
    ];

    shellHook = ''
        export JULIA_DEPOT_PATH=./juliapkgs
    '';
}

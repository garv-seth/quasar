{pkgs}: {
  deps = [
    pkgs.chromedriver
    pkgs.chromium
    pkgs.geckodriver
    pkgs.libxcrypt
    pkgs.spdlog
    pkgs.nlohmann_json
    pkgs.muparserx
    pkgs.fmt
    pkgs.catch2
    pkgs.glib
    pkgs.gtk3
    pkgs.xorg.libXrandr
    pkgs.xorg.libXext
    pkgs.xorg.libX11
    pkgs.at-spi2-core
    pkgs.udev
    pkgs.dbus
    pkgs.alsa-lib
    pkgs.cairo
    pkgs.pango
    pkgs.xorg.libXfixes
    pkgs.xorg.libXdamage
    pkgs.xorg.libXcomposite
    pkgs.xorg.libxcb
    pkgs.expat
    pkgs.cups
    pkgs.at-spi2-atk
    pkgs.atk
    pkgs.nss
    pkgs.bash
    pkgs.postgresql
    pkgs.playwright-driver
    pkgs.gitFull
    pkgs.ffmpeg-full
    pkgs.glibcLocales
  ];
}

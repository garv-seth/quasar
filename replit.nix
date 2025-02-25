{pkgs}: {
  deps = [
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

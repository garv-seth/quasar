{pkgs}: {
  deps = [
    pkgs.bash
    pkgs.postgresql
    pkgs.playwright-driver
    pkgs.gitFull
    pkgs.ffmpeg-full
    pkgs.glibcLocales
  ];
}

# Reference data

This folder is expected by `specred.utils` (arc-lamp line lists) and
`specred.utils.stdfile` (standard-star flux tables), but the actual
reference files were not part of the uploaded scripts, so they are not
included in this repository yet.

Expected layout:

```
resources/
├── arc/
│   └── HgAr_neon.dat        # arc lamp reference wavelength/flux table
│                            # (referenced by fileinfo.yaml -> arc_lamp.name)
└── onedstar/
    └── <catalog_folder>/
        └── <starname>.dat   # standard-star flux tables, e.g. from the
                              # IRAF onedstd distribution (feige66.dat, etc.)
```

Before publishing, add:
1. The arc-lamp line list(s) your instrument uses, in `arc/`.
2. The standard-star flux calibration table(s) you use, in
   `onedstar/<catalog>/`, matching whatever `standard_star.name` you
   put in your `fileinfo.yaml`.

If these come from a public catalog (e.g. IRAF `onedstds`), note the
source/license in this file rather than silently vendoring them.

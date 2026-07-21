# PEPS paper datasets

This directory contains only code, frozen manifests, checksums, and provenance.
Raw images, texture maps, meshes, and generated SDF volumes stay below ignored
paths and must not be committed.

## Acquire and verify

```bash
python data/download.py list all
python data/download.py fetch kodak
python data/download.py fetch textures
python data/download.py fetch sdf

python data/download.py verify kodak
python data/download.py verify textures
python data/download.py verify sdf --asset lucy \
  --asset thai-statue --asset armadillo
```

Downloads are written atomically. Existing files are accepted only after byte
size and checksum validation. ZIP and tar members are selected by exact manifest
name; the downloader does not use unrestricted `extractall`.

The manifests freeze:

- all 24 original-orientation Kodak PCD0992 PNGs;
- the paper appendix's 18 named 4K texture sets (13 Poly Haven and 5 ambientCG);
- Lucy, Pitted Stonefish, Thai Statue, and Armadillo meshes;
- source URL, local path, map semantic, color space, resolution, file size,
  checksum, license, and any access restriction.

## Texture contract

Use the strict dynamic loader, not the legacy fixed 9-channel teaching bundle:

```python
from data.manifest import load_texture_set

loaded = load_texture_set("paving-stones-070")
target = loaded.tensor  # H x W x (3 * number_of_maps)
```

Every listed map is required. Grayscale data maps are replicated to RGB because
the paper defines a texture set target in `R^(3k)`; an absent map is never
replaced by a constant. DIFF maps remain sRGB-encoded targets, while data maps
are linear. Alpha is not a Table 2 category and is discarded.

OpenGL normals are decoded from `[0,1]` to vectors in `[-1,1]`, bilinearly
filtered, normalized to unit length, then re-encoded. This applies at native
resolution too, removing quantization-induced length error.

The paper publishes asset names and eight aggregate map categories, but not a
per-file/channel manifest. The frozen selection therefore states one explicit
reproduction assumption:

- Poly Haven uses packed ARM, DIFF/color variants, OpenGL normal, displacement,
  and specular when available. Separate AO/rough/metal maps that duplicate ARM,
  plus masks, opacity, bump, and `rough_ao`, are excluded.
- ambientCG uses AO, Color, Displacement, Metalness when supplied, NormalGL, and
  Roughness from each lossless `4K-PNG` archive.

Do not silently change this selection. A different selection is a dataset
ablation and needs a separate manifest/result namespace.

## Pitted Stonefish access

The canonical paper asset is Sketchfab UID
`0cdc3d1419384fd78fd952dc251a3169`: the 10.5M-face CT scan marked
Academic-only and licensed CC BY-NC-SA 4.0. The public metadata endpoint is
verified without credentials, but download requires account authorization.

Use one of the following environment variables; never put a token in a command,
manifest, `.env` file committed to git, log, or notebook:

```bash
export SKETCHFAB_OAUTH_TOKEN='...'
# or: export SKETCHFAB_API_TOKEN='...'
python data/download.py fetch sdf --asset pitted-stonefish
```

For a browser/manual download, place the authorized file at
`data/raw/sdf/meshes/pitted_stonefish.glb`, then run the same command without a
token. It records a checksum in a git-ignored local acquisition receipt.

The CC0 surface scan UID `7db8da6be6434143af1bc618a4062fb3` is recorded as a
non-canonical substitution. It is not the cited CT mesh and must not be used in
canonical paper tables.

## Mesh to 512³ SDF

The default command follows `sdf.json` exactly:

```bash
python -m pip install -r data/requirements-preprocess.txt
python data/preprocess_sdf.py armadillo
```

It performs one shared normalization for all assets:

1. subtract the axis-aligned bounding-box center;
2. isotropically scale the largest extent to span `[-1,1]`;
3. build an Open3D BVH over every source triangle and compute exact
   point-to-triangle distance;
4. classify inside/outside with eleven-ray parity voting, which avoids the
   isolated sign failures observed on Lucy's documented topology defects;
5. query an inclusive `512 x 512 x 512` grid in `zyx` storage order;
6. store float32 distance in centered-domain units, negative inside;
7. write atomically and emit a checksum-bearing provenance JSON.

The implementation queries `Open3D RaycastingScene` in z slabs and writes
through a NumPy memmap, so it does not allocate all query coordinates at once.
There is no automatic sign/method fallback. Legacy sampled/scan methods remain
available only through explicit non-canonical overrides.

A small algorithm smoke run must be explicitly marked non-canonical:

```bash
python data/preprocess_sdf.py armadillo \
  --resolution 32 \
  --allow-protocol-override
```

The paper says its volumes were produced by an unreleased C++/HIP application.
The checked-in exact-triangle Open3D protocol is reproducible and auditable, but
is not a bit-exact reconstruction of that converter. Preserve this limitation
in result reports.

## Licenses

- Poly Haven and ambientCG texture assets are CC0.
- Kodak PCD0992 is not CC0. The sampler grants electronic-imaging uses to
  authorized holders and generally requires photographer credit for
  reproductions; consult the URL in `kodak.json`.
- Stanford meshes are available for research/non-commercial use with source
  acknowledgement. Lucy and Thai Statue also carry good-taste/use requests
  stated in `sdf.json`.
- The canonical Stonefish CT model is CC BY-NC-SA 4.0 and marked Academic-only.

`data/provenance/source-verification.json` records what was actually checked
without embedding raw data, signed download URLs, or credentials.

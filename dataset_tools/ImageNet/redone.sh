src="."
dst="."
mkdir -p "$dst"

find "$src" -type f -name "*.jpg" -print0 |
while IFS= read -r -d '' f; do
  folder="$(basename "$(dirname "$f")")"
  file="$(basename "$f")"
  cp -n "$f" "$dst/${folder}_${file}"
done

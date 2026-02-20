#!/usr/bin/env python3
"""
Delete ImageNet subfolders that are NOT in the allowlist you provided.

What it does:
- Looks at *immediate* subdirectories of IMAGENET_DIR
- Deletes any folder whose name is NOT in the allowlist
- Keeps allowlisted folders
- Supports --dry-run

Usage:
  # Preview what would be deleted
  python prune_imagenet_folders.py ./data/imagenet --dry-run

  # Actually delete
  python prune_imagenet_folders.py ./data/imagenet

Notes:
- This does NOT touch files at the top level (only folders).
- If your classes are inside a nested directory (e.g. ./data/imagenet/extracted/train),
  point the script at THAT directory instead.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ALLOWED_DIRS = {
    "beaver",
    "angora",
    "ant",
    "cleaver",
    "alligator_lizard",
    "chainlink_fence",
    "bedlington_terrier",
    "analog_clock",
    "corkscrew",
    "bobsled",
    "breakwater",
    "black_swan",
    "clumber",
    "black_widow",
    "chiton",
    "beach_wagon",
    "croquet_ball",
    "bassinet",
    "black_and_tan_coonhound",
    "coho",
    "clog",
    "buckeye",
    "bow_tie",
    "chambered_nautilus",
    "australian_terrier",
    "consomme",
    "bath_towel",
    "crate",
    "bell_cote",
    "cicada",
    "bison",
    "carton",
    "broom",
    "coucal",
    "cougar",
    "combination_lock",
    "bittern",
    "cock",
    "balloon",
    "chesapeake_bay_retriever",
    "basketball",
    "brittany_spaniel",
    "bannister",
    "bagel",
    "anemone_fish",
    "bulbul",
    "beagle",
    "birdhouse",
    "black_and_gold_garden_spider",
    "ambulance",
    "conch",
    "cliff",
    "cardigan",
    "barometer",
    "border_collie",
    "ballplayer",
    "bucket",
    "crayfish",
    "african_chameleon",
    "butternut_squash",
    "brain_coral",
    "cowboy_boot",
    "bikini",
    "baseball",
    "boa_constrictor",
    "binoculars",
    "corn",
    "baboon",
    "barbershop",
    "affenpinscher",
    "arabian_camel",
    "axolotl",
    "cash_machine",
    "bassoon",
    "candle",
    "barbell",
    "agaric",
    "border_terrier",
    "cheeseburger",
    "brass",
    "barrel",
    "cauliflower",
    "brown_bear",
    "bathtub",
    "black_grouse",
    "assault_rifle",
    "barracouta",
    "cassette",
    "bluetick",
    "container_ship",
    "coil",
    "borzoi",
    "appenzeller",
    "banded_gecko",
    "brassiere",
    "bearskin",
    "bookcase",
    "alp",
    "car_wheel",
    "car_mirror",
    "bakery",
    "american_egret",
    "centipede",
    "banjo",
    "barn",
    "american_black_bear",
    "crib",
    "cocktail_shaker",
    "bow",
    "boston_bull",
    "flat",
    "airliner",
    "basset",
    "breastplate",
    "african_grey",
    "cellular_telephone",
    "china_cabinet",
    "barn_spider",
    "arctic_fox",
    "book_jacket",
    "burrito",
    "african_crocodile",
    "canoe",
    "boathouse",
    "can_opener",
    "chow",
    "chocolate_sauce",
    "caldron",
    "backpack",
    "common_newt",
    "crash_helmet",
    "boxer",
    "beacon",
    "church",
    "bighorn",
    "val_cache_256_mmap",
    "box_turtle",
    "afghan_hound",
    "cardigan_welsh_corgi",
    "bolo_tie",
    "albatross",
    "cowboy_hat",
    "artichoke",
    "bee",
    "crane_bird",
    "african_hunting_dog",
    "cockroach",
    "apron",
    "carousel",
    "american_staffordshire_terrier",
    "american_coot",
    "barrow",
    "bulletproof_vest",
    "bloodhound",
    "balance_beam",
    "cello",
    "airship",
    "cannon",
    "ashcan",
    "banana",
    "cloak",
    "airedale",
    "catamaran",
    "acorn_squash",
    "confectionery",
    "admiral",
    "bottlecap",
    "colobus",
    "amphibian",
    "bubble",
    "bullet_train",
    "coffee_mug",
    "castle",
    "bookshop",
    "black_footed_ferret",
    "test_cache_256_mmap",
    "cabbage_butterfly",
    "christmas_stocking",
    "cassette_player",
    "band_aid",
    "chime",
    "ballpoint",
    "comic_book",
    "chain",
    "bell_pepper",
    "train_cache_256_mmap",
    "bernese_mountain_dog",
    "bathing_cap",
    "computer_keyboard",
    "cairn",
    "splits",
    "african_elephant",
    "cricket",
    "coyote",
    "carbonara",
    "bonnet",
    "bustard",
    "chihuahua",
    "crane",
    "american_lobster",
    "american_alligator",
    "black_stork",
    "chimpanzee",
    "capuchin",
    "chiffonier",
    "apiary",
    "chain_mail",
    "briard",
    "chest",
    "cab",
    "bouvier_des_flandres",
    "cliff_dwelling",
    "cradle",
    "blenheim_spaniel",
    "beer_bottle",
    "badger",
    "barber_chair",
    "chain_saw",
    "coral_reef",
    "american_chameleon",
    "bullfrog",
    "cinema",
    "agama",
    "coral_fungus",
    "bicycle_built_for_two",
    "coffeepot",
    "carpenter_s_kit",
    "chickadee",
    "crock_pot",
    "basenji",
    "bee_eater",
    "bull_mastiff",
    "cocker_spaniel",
    "collie",
    "armadillo",
    "altar",
    "beaker",
    "broccoli",
    "bald_eagle",
    "binder",
    "buckle",
    "acoustic_guitar",
    "cardoon",
    "butcher_shop",
    "common_iguana",
    "bib",
    "aircraft_carrier",
    "cd_player",
    "convertible",
    "brabancon_griffon",
    "cheetah",
    "bolete",
    "cornet",
    "beer_glass",
    "brambling",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("imagenet_dir", type=Path, help="Directory whose immediate subfolders will be pruned")
    ap.add_argument("--dry-run", action="store_true", help="Print folders that would be deleted (no deletion)")
    ap.add_argument("--keep-unknown-files", action="store_true",
                    help="Do nothing to files (default). Present for clarity; files are never deleted.")
    args = ap.parse_args()

    root: Path = args.imagenet_dir
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    subdirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)

    to_delete = [d for d in subdirs if d.name not in ALLOWED_DIRS]
    to_keep = [d for d in subdirs if d.name in ALLOWED_DIRS]

    print(f"Target: {root.resolve()}")
    print(f"Found subfolders: {len(subdirs)} | keep: {len(to_keep)} | delete: {len(to_delete)}")

    if args.dry_run:
        if to_delete:
            print("\n[DRY RUN] Would delete:")
            for d in to_delete:
                print(f"  {d.name}/")
        else:
            print("\n[DRY RUN] Nothing to delete.")
        return 0

    # Delete
    deleted = 0
    for d in to_delete:
        shutil.rmtree(d)
        deleted += 1

    print(f"\nDeleted {deleted} folders not in allowlist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
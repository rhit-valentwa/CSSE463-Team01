#!/usr/bin/env python3
"""
Unzip an archive, then delete every folder in the output directory that is NOT
in the allowlist below.

Usage:
  python prune_imagenet_dirs.py /path/to/imagenet-256.zip /path/to/output_dir

Example:
  python prune_imagenet_dirs.py ~/Downloads/imagenet-256.zip ~/Downloads/imagenet-256
"""

import argparse
import os
import shutil
import sys
import zipfile
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


def unzip(zip_path: Path, out_dir: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Basic zip-slip protection: ensure extracted paths stay within out_dir.
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            dest = out_dir / member.filename
            resolved = dest.resolve()
            if not str(resolved).startswith(str(out_dir.resolve()) + os.sep) and resolved != out_dir.resolve():
                raise RuntimeError(f"Unsafe path in zip (zip-slip attempt?): {member.filename}")
        zf.extractall(out_dir)


def prune_folders(out_dir: Path, allowed: set[str]) -> tuple[int, list[str]]:
    if not out_dir.exists():
        raise FileNotFoundError(f"Output dir not found: {out_dir}")

    deleted = []
    count = 0

    for p in out_dir.iterdir():
        if p.is_dir():
            name = p.name
            if name not in allowed:
                shutil.rmtree(p)
                deleted.append(name)
                count += 1

    return count, sorted(deleted)


def main() -> int:
    ap = argparse.ArgumentParser(description="Unzip archive then delete folders not in allowlist.")
    ap.add_argument("zip_path", type=Path, help="Path to .zip archive")
    ap.add_argument("out_dir", type=Path, help="Directory to extract into (and prune)")
    ap.add_argument("--skip-unzip", action="store_true", help="Do not unzip; only prune existing out_dir")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be deleted, but don't delete")
    args = ap.parse_args()

    if not args.skip_unzip:
        print(f"Unzipping: {args.zip_path} -> {args.out_dir}")
        unzip(args.zip_path, args.out_dir)

    # Determine folders to delete first (so dry-run works cleanly)
    to_delete = []
    for p in args.out_dir.iterdir():
        if p.is_dir() and p.name not in ALLOWED_DIRS:
            to_delete.append(p)

    if args.dry_run:
        print("Dry run: would delete these folders:")
        for p in sorted(to_delete, key=lambda x: x.name):
            print(f"  {p.name}")
        print(f"Total: {len(to_delete)} folders would be deleted.")
        return 0

    n, deleted = prune_folders(args.out_dir, ALLOWED_DIRS)
    print(f"Deleted {n} folders.")
    if deleted:
        print("Deleted:")
        for name in deleted:
            print(f"  {name}")
    else:
        print("Nothing to delete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
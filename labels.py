# labels.py

# The 20 ImageNet classes chosen for this hackathon.
#
# HF_INDEX is the ImageNet-1K integer label used by Hugging Face.
# LOCAL_INDEX is the 0–19 hackathon label used by your models.

HF_INDEX_TO_NAME = {
    # Aquatic animals
    1: "goldfish",
    107: "jellyfish",

    # Birds
    22: "bald_eagle",
    96: "toucan",

    # Land mammals
    292: "tiger",
    386: "african_elephant",

    # Structures / vehicles
    483: "castle",
    510: "container_ship",
    817: "sports_car",

    # Aircraft
    404: "airliner",
    417: "balloon",

    # Food
    951: "lemon",
    963: "pizza",

    # Technology
    487: "mobile_phone",
    620: "laptop",

    # Musical instruments
    402: "acoustic_guitar",
    566: "french_horn",

    # Plants / natural objects
    947: "mushroom",
    985: "daisy",          # replacement for sunflower; sunflower is not in ImageNet-1K

    # Structures
    437: "lighthouse",
}

ORDERED_HF_INDICES = sorted(HF_INDEX_TO_NAME.keys())

IDX_TO_HF_INDEX = {
    local_idx: hf_idx
    for local_idx, hf_idx in enumerate(ORDERED_HF_INDICES)
}

HF_INDEX_TO_IDX = {
    hf_idx: local_idx
    for local_idx, hf_idx in IDX_TO_HF_INDEX.items()
}

IDX_TO_NAME = {
    local_idx: HF_INDEX_TO_NAME[hf_idx]
    for local_idx, hf_idx in IDX_TO_HF_INDEX.items()
}

TARGET_HF_INDICES = set(HF_INDEX_TO_NAME.keys())


if __name__ == "__main__":
    print(f"{'Local':<7} {'HF index':<9} {'Class'}")
    print("-" * 35)
    for local_idx, hf_idx in IDX_TO_HF_INDEX.items():
        print(f"{local_idx:<7} {hf_idx:<9} {HF_INDEX_TO_NAME[hf_idx]}")
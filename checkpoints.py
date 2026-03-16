# ============================================================
# Checkpoint Registry
# ============================================================
# These dictionaries define the checkpoint paths used for
# evaluation. Each key corresponds to a downstream task (DT),
# (MT), or (LT) training setup.
#
# Structure:
#   {
#       "<TASK_NAME>": [
#           "<checkpoint_path_1>",
#           "<checkpoint_path_2>",
#           "<checkpoint_path_3>",
#       ]
#   }
#
# Notes:
# - Each task contains 3 independently trained runs.
# - Checkpoints follow the naming convention:
#     <TASK>_eca_nfnet_l1_<YYYY-MM-DD>_<HHMMSS>
# - These are typically averaged during evaluation.
# ============================================================


# ------------------------------------------------------------
# Downstream Task (DT) checkpoints
# ------------------------------------------------------------
# Each key corresponds to a specific downstream dataset.
# Used when evaluating task-specific fine-tuned models.

DT = {
    "HSN": ["ckpts/DT/HSN/HSN_eca_nfnet_l1_2025-10-20_112131",
            "ckpts/DT/HSN/HSN_eca_nfnet_l1_2025-10-20_113316",
            "ckpts/DT/HSN/HSN_eca_nfnet_l1_2025-10-20_114501"],

    "POW": ["ckpts/DT/POW/POW_eca_nfnet_l1_2025-10-11_143357",
            "ckpts/DT/POW/POW_eca_nfnet_l1_2025-10-11_145846",
            "ckpts/DT/POW/POW_eca_nfnet_l1_2025-10-11_152335"],

    "SNE": ["ckpts/DT/SNE/SNE_eca_nfnet_l1_2025-10-12_131606",
            "ckpts/DT/SNE/SNE_eca_nfnet_l1_2025-10-12_135017",
            "ckpts/DT/SNE/SNE_eca_nfnet_l1_2025-10-12_142429"],

    "PER": ["ckpts/DT/PER/PER_eca_nfnet_l1_2025-10-12_194055",
            "ckpts/DT/PER/PER_eca_nfnet_l1_2025-10-12_201638",
            "ckpts/DT/PER/PER_eca_nfnet_l1_2025-10-12_204942"],

    "NES": ["ckpts/DT/NES/NES_eca_nfnet_l1_2025-10-13_083702",
            "ckpts/DT/NES/NES_eca_nfnet_l1_2025-10-13_091658",
            "ckpts/DT/NES/NES_eca_nfnet_l1_2025-10-13_095515"],

    "UHH": ["ckpts/DT/UHH/UHH_eca_nfnet_l1_2025-10-13_130012",
            "ckpts/DT/UHH/UHH_eca_nfnet_l1_2025-10-13_132434",
            "ckpts/DT/UHH/UHH_eca_nfnet_l1_2025-10-13_134708"],

    "NBP": ["ckpts/DT/NBP/NBP_eca_nfnet_l1_2025-10-13_143637",
            "ckpts/DT/NBP/NBP_eca_nfnet_l1_2025-10-13_151856",
            "ckpts/DT/NBP/NBP_eca_nfnet_l1_2025-10-13_160131"],

    "SSW": ["ckpts/DT/SSW/SSW_eca_nfnet_l1_2025-10-11_215626",
            "ckpts/DT/SSW/SSW_eca_nfnet_l1_2025-10-11_203307",
            "ckpts/DT/SSW/SSW_eca_nfnet_l1_2025-10-11_190939"],
}

MT = {"ALL": ["ckpts/MT/MT_eca_nfnet_l1_2025-11-25_151907",
              "ckpts/MT/MT_eca_nfnet_l1_2025-11-25_214315",
              "ckpts/MT/MT_eca_nfnet_l1_2025-11-26_010815"]
      }

LT = {"ALL": ["ckpts/LT/LT_eca_nfnet_l1_2025-11-25_043730",
              "ckpts/LT/LT_eca_nfnet_l1_2025-11-25_060339",
              "ckpts/LT/LT_eca_nfnet_l1_2025-11-24_180849"]
      }

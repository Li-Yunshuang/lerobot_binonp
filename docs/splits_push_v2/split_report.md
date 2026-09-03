# Push dataset object split

- grouping key: base mesh (orientation prefix stripped) -- 61 meshes over 2115 episodes
- seed 0, target held-out fraction 0.2
- **train**: 48 meshes / 1643 episodes
- **held-out**: 13 meshes / 472 episodes

## Held-out meshes

| mesh | group | episodes | expert success |
|---|---|---|---|
| BIA_Cordon_Bleu_White_Porcelain_Utensil_Holder_900028 | non-convex | 26 | 26/30 |
| COAST_GUARD_BOAT | non-convex | 11 | 11/30 |
| Cole_Hardware_Butter_Dish_Square_Red | non-convex | 29 | 29/30 |
| Crayola_Washable_Sidewalk_Chalk_16_pack | rotated | 99 | 99/120 |
| Crunch_Girl_Scouts_Candy_Bars_Peanut_Butter_Creme_78_oz_box | round | 57 | 57/60 |
| Dell_Ink_Cartridge | non-convex | 45 | 45/60 |
| Granimals_20_Wooden_ABC_Blocks_Wagon_85VdSftGsLi | rotated | 59 | 59/60 |
| Lenovo_Yoga_2_11 | rotated | 12 | 12/60 |
| Nescafe_Momento_Mocha_Specialty_Coffee_Mix_8_ct | rotated | 27 | 27/30 |
| Philips_EcoVantage_43_W_Light_Bulbs_Natural_Light_2_pack | round | 29 | 29/30 |
| Phillips_Caplets_Size_24 | rotated | 24 | 24/30 |
| TriStar_Products_PPC_Power_Pressure_Cooker_XL_in_Black | round | 14 | 14/30 |
| frustum_35x25x15_15deg | frustum | 40 | 40/60 |

## Per-group balance

| group | train meshes | held-out meshes |
|---|---|---|
| frustum | 1 | 1 |
| non-convex | 14 | 4 |
| rotated | 22 | 5 |
| round | 11 | 3 |

## Curated evaluation sets (expert >= 75%)

- held-out: **14** of 21 asset paths, expert 91.9%
- in-domain: **14** paths sampled from 49 eligible training paths, expert 90.7%
- dropped from held-out: COAST_GUARD_BOAT, Lenovo_Yoga_2_11, neg_x_down__Lenovo_Yoga_2_11, pos_z_down__Dell_Ink_Cartridge, pos_z_down__frustum_35x25x15_15deg, Crayola_Washable_Sidewalk_Chalk_16_pack, TriStar_Products_PPC_Power_Pressure_Cooker_XL_in_Black

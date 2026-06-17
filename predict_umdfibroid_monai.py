python predict_umdfibroid_monai.py   --input-dir /media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UMD/nnUNet_testprep_umd/imagesTs   --output-dir /media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UMD/outputs_ensemble_ext   --plans /media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UMD/nnUNet_raw_data_base/nnUNet_preprocessed/Dataset004_UMD/nnUNetPlans.json   --dataset-json /media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UMD/nnUNet_raw_data_base/nnUNet_preprocessed/Dataset004_UMD/dataset.json   --configuration 3d_fullres   --model ensemble  --runs-root /media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UterusSegBenchmark/UterusSegBenchmark/runs/umd_multiclass  --folds all   --prefer best   --task multiclass   --device cuda   --save-probabilities
<frozen importlib._bootstrap_external>:1241: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
[dataset.json] inferred num_classes=5
[nnunetv2] configuration=3d_fullres spacing_zyx=(4.400000095367432, 0.4838709533214569, 0.4838709533214569) patch_size_zyx=(16, 320, 320)
[predict] Ensemble members: unet3d, swinunetr, mednext
[predict] Using checkpoints for unet3d:
 - /media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UterusSegBenchmark/UterusSegBenchmark/runs/umd_multiclass/unet3d/fold_00/best_unet3d_fold00.pt
 - /media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UterusSegBenchmark/UterusSegBenchmark/runs/umd_multiclass/unet3d/fold_01/best_unet3d_fold01.pt
 - /media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UterusSegBenchmark/UterusSegBenchmark/runs/umd_multiclass/unet3d/fold_02/best_unet3d_fold02.pt
 - /media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UterusSegBenchmark/UterusSegBenchmark/runs/umd_multiclass/unet3d/fold_03/best_unet3d_fold03.pt
 - /media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UterusSegBenchmark/UterusSegBenchmark/runs/umd_multiclass/unet3d/fold_04/best_unet3d_fold04.pt
[load] best_unet3d_fold00.pt roi_from_ckpt=True roi_use=(16, 320, 320) spacing_from_ckpt=True spacing_use=(4.400000095367432, 0.4838709533214569, 0.4838709533214569)
[load] best_unet3d_fold01.pt roi_from_ckpt=True roi_use=(16, 320, 320) spacing_from_ckpt=True spacing_use=(4.400000095367432, 0.4838709533214569, 0.4838709533214569)
[load] best_unet3d_fold02.pt roi_from_ckpt=True roi_use=(16, 320, 320) spacing_from_ckpt=True spacing_use=(4.400000095367432, 0.4838709533214569, 0.4838709533214569)
[load] best_unet3d_fold03.pt roi_from_ckpt=True roi_use=(16, 320, 320) spacing_from_ckpt=True spacing_use=(4.400000095367432, 0.4838709533214569, 0.4838709533214569)
[load] best_unet3d_fold04.pt roi_from_ckpt=True roi_use=(16, 320, 320) spacing_from_ckpt=True spacing_use=(4.400000095367432, 0.4838709533214569, 0.4838709533214569)
[predict] Using checkpoints for swinunetr:
 - /media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UterusSegBenchmark/UterusSegBenchmark/runs/umd_multiclass/swinunetr/fold_00/best_swinunetr_fold00.pt
 - /media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UterusSegBenchmark/UterusSegBenchmark/runs/umd_multiclass/swinunetr/fold_01/best_swinunetr_fold01.pt
 - /media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UterusSegBenchmark/UterusSegBenchmark/runs/umd_multiclass/swinunetr/fold_02/best_swinunetr_fold02.pt
 - /media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UterusSegBenchmark/UterusSegBenchmark/runs/umd_multiclass/swinunetr/fold_03/best_swinunetr_fold03.pt
 - /media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UterusSegBenchmark/UterusSegBenchmark/runs/umd_multiclass/swinunetr/fold_04/best_swinunetr_fold04.pt
[load] best_swinunetr_fold00.pt roi_from_ckpt=True roi_use=(32, 320, 320) spacing_from_ckpt=True spacing_use=(4.400000095367432, 0.4838709533214569, 0.4838709533214569)
[swinunetr] roi=(32, 352, 352) patch_size=(1, 2, 2) window_size=(4, 7, 7) fs=12 depths=(2, 2, 2, 1) heads=(3, 6, 12, 24)
[load] Dropping 15 unexpected/mismatched keys (showing up to 12):
   - swinViT.patch_embed.proj.weight
   - swinViT.layers1.0.blocks.0.attn.relative_position_bias_table
   - swinViT.layers1.0.blocks.0.attn.relative_position_index
   - swinViT.layers1.0.blocks.1.attn.relative_position_bias_table
   - swinViT.layers1.0.blocks.1.attn.relative_position_index
   - swinViT.layers2.0.blocks.0.attn.relative_position_bias_table
   - swinViT.layers2.0.blocks.0.attn.relative_position_index
   - swinViT.layers2.0.blocks.1.attn.relative_position_bias_table
   - swinViT.layers2.0.blocks.1.attn.relative_position_index
   - swinViT.layers3.0.blocks.0.attn.relative_position_bias_table
   - swinViT.layers3.0.blocks.0.attn.relative_position_index
   - swinViT.layers3.0.blocks.1.attn.relative_position_bias_table
   ... (3 more)
Traceback (most recent call last):
  File "/media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UterusSegBenchmark/UterusSegBenchmark/predict_umdfibroid_monai.py", line 1106, in <module>
    main()
  File "/media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UterusSegBenchmark/UterusSegBenchmark/predict_umdfibroid_monai.py", line 986, in main
    member_models, member_roi, member_spacing = load_models_from_checkpoints(
                                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/daniel/b6eaf548-4cbb-4781-8be0-fea0091c0087/UterusSegBenchmark/UterusSegBenchmark/predict_umdfibroid_monai.py", line 791, in load_models_from_checkpoints
    raise RuntimeError(
RuntimeError: best_swinunetr_fold00.pt would drop 15 mismatched state_dict key(s) for swinunetr. This indicates the prediction model was not rebuilt exactly like training.
fennet mhc embed_proteins --fasta test.fasta  --save_pkl test.pkl --load_model_hla ../resource/HLA_model_v0819.pt

fennet mhc embed_peptides_tsv --tsv 0813_100k_peptides.tsv  --save_pkl 0813_100k_peptides.pkl --load_model_pept ../resource/pept_model_v0819.pt

fennet mhc embed_peptides_fasta --fasta ../resource/uniprotkb_UP000005640_AND_reviewed_true_2024_03_01.fasta --save_pkl human_9mers.pkl --min_peptide_length 9 --max_peptide_length 9 --load_model_pept ../resource/pept_model_v0819.pt

fennet mhc predict_binding_for_MHC --peptide_pkl 0813_100k_peptides.pkl --protein_pkl ../resource/hla_v0819_embeds.pkl --alleles A01_01,B07_02,C14_02 --human_protein_fasta ../resource/uniprotkb_UP000005640_AND_reviewed_true_2024_03_01.fasta --out-folder ./ --load_model_hla ../resource/HLA_model_v0819.pt --load_model_pept ../resource/pept_model_v0819.pt

fennet mhc predict_binding_for_epitope --peptide_pkl 0813_100k_peptides.pkl --protein_pkl ../resource/hla_v0819_embeds.pkl --human_protein_fasta ../resource/uniprotkb_UP000005640_AND_reviewed_true_2024_03_01.fasta --out-folder ./ --load_model_hla ../resource/HLA_model_v0819.pt --load_model_pept ../resource/pept_model_v0819.pt

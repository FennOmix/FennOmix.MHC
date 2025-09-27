# import os

# from click.testing import CliRunner

# from fennomix_mhc import cli

# TEST_PEPTIDE_TSV = os.path.abspath("./test_data/test_peptides.tsv")
# TEST_MHC_FASTA = os.path.abspath("./test_data/test_MHC_proteins.fasta")
# TEST_PEPTIDE_FASTA = os.path.abspath("./test_data/test_peptides.fasta")
# OUT_DIR = os.path.abspath("./nogit")

# def test_cli_embed_peptides():
#     runner = CliRunner()
#     result = runner.invoke(cli, ['embed-proteins', '--fasta', TEST_MHC_FASTA, '--out-folder', OUT_DIR])
#     assert result.exit_code == 0

# def test_cli_embed_peptides_tsv():
#     runner = CliRunner()
#     result = runner.invoke(cli, ['embed-peptides', '--peptide-file', TEST_PEPTIDE_TSV, '--out-folder', OUT_DIR])
#     assert result.exit_code == 0

# def test_cli_embed_peptides_fasta():
#     runner = CliRunner()
#     result = runner.invoke(cli, ['embed-peptides', '--peptide-file', TEST_PEPTIDE_FASTA, '--out-folder', OUT_DIR])
#     assert result.exit_code == 0

# def test_cli_predict_epitopes_for_mhc():
#     runner = CliRunner()
#     result = runner.invoke(cli, ['predict-epitopes-for-mhc', '--peptide-file',
#                                  TEST_PEPTIDE_TSV, '--alleles' 'A02_01', '--out-folder', OUT_DIR])
#     assert result.exit_code == 0

# def test_cli_predict_mhc_binders_for_epitopes():
#     runner = CliRunner()
#     result = runner.invoke(cli, ['predict-mhc-binders-for-epitopes', '--peptide-file', TEST_PEPTIDE_TSV,
#                                  '--out-folder', OUT_DIR])
#     assert result.exit_code == 0

# # def test_cli_deconvolute_peptides():
# #     runner = CliRunner()
# #     result = runner.invoke(cli, ['deconvolute-peptides', '--peptide-file', TEST_PEPTIDE_TSV,
# #                                  '--n-centroids', '2', '--out-folder', OUT_DIR])
# #     assert result.exit_code == 0

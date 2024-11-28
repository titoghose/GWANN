# seeds1="0,142,172,281,3589,363,37,4250,675,7282,893,89,937"
seeds1="56,530,265"

echo "Running for Black population"
python non_white_validation.py --pop "Black" --seeds $seeds1 --gpu 0

echo "Running for Asian population"
python non_white_validation.py --pop "Asian" --seeds $seeds1 --gpu 1

echo "Running Dummy for Black population"
python non_white_dummy_genes.py --pop "Black" --seeds $seeds1 --gpu 0

echo "Running Dummy for Asian population"
python non_white_dummy_genes.py --pop "Asian" --seeds $seeds1 --gpu 1
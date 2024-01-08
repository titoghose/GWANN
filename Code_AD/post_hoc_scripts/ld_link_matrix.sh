out_file=$1
snp_list=$2
curl \
    -k \
    -H "Content-Type: application/json" \
    -X POST \
    -o $out_file \
    -d '{"snps":"'$snp_list'", "pop": "GBR", "r2_d": "r2", "genome_build": "grch37"}' \
    'https://ldlink.nih.gov/LDlinkRest/ldmatrix?token=81b0b1b59294'

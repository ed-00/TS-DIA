utils/combine_data.sh data/simu/data/train_combined \
  data/simu/data/train_clean_5_ns1_beta2_100000 \
  data/simu/data/train_clean_5_ns2_beta2_100000 \
  data/simu/data/train_clean_5_ns3_beta5_100000 \
  data/simu/data/train_clean_5_ns4_beta9_100000 \
  data/simu/data/train_clean_5_ns5_beta13_100000

utils/combine_data.sh data/simu/data/dev_combined \
  data/simu/data/dev_clean_2_ns1_beta2_100000 \
  data/simu/data/dev_clean_2_ns2_beta2_500 \
  data/simu/data/dev_clean_2_ns2_beta3_500 \
  data/simu/data/dev_clean_2_ns2_beta5_500 \
  data/simu/data/dev_clean_2_ns3_beta5_500 \
  data/simu/data/dev_clean_2_ns3_beta7_500 \
  data/simu/data/dev_clean_2_ns3_beta11_500 \
  data/simu/data/dev_clean_2_ns4_beta9_500 \
  data/simu/data/dev_clean_2_ns5_beta13_500


print_success "All datasets combined successfully"

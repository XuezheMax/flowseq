#!/bin/bash

dir=$1
lang=$2
num_operations=$3

train_src=${dir}/train.en
train_tgt=${dir}/train.${lang}
dev_src=${dir}/dev.en
dev_tgt=${dir}/dev.${lang}
test_src=${dir}/test.en
test_tgt=${dir}/test.${lang}

code_dir=${dir}/"bpe-codes"
if [[ ! -e ${code_dir} ]]; then
    mkdir -p ${code_dir}
fi

codes_file="${code_dir}/joint-bpe-$((num_operations / 1000))k.codes"
vocab_file="${code_dir}/sep_vocab"

# this bash file has to be run under the directory of each dataset
write_dir="./joint-bpe-$((num_operations / 1000))k"
echo ${write_dir}
if [[ ! -e ${write_dir} ]]; then
    mkdir -p ${write_dir}
fi

train_src_out="${write_dir}/train.en"
train_tgt_out="${write_dir}/train.${lang}"
dev_src_out="${write_dir}/dev.en"
dev_tgt_out="${write_dir}/dev.${lang}"
test_src_out="${write_dir}/test.en"
test_tgt_out="${write_dir}/test.${lang}"

cat ${train_src} ${train_tgt} | subword-nmt learn-bpe -s ${num_operations} -o ${codes_file}

subword-nmt apply-bpe -c ${codes_file} < ${train_src} | subword-nmt get-vocab > "${vocab_file}.en"
subword-nmt apply-bpe -c ${codes_file} < ${train_tgt} | subword-nmt get-vocab > "${vocab_file}.${lang}"

subword-nmt apply-bpe -c ${codes_file} --vocabulary "${vocab_file}.en" --vocabulary-threshold 2 < ${train_src} > ${train_src_out}
subword-nmt apply-bpe -c ${codes_file} --vocabulary "${vocab_file}.${lang}" --vocabulary-threshold 2 < ${train_tgt} > ${train_tgt_out}

subword-nmt apply-bpe -c ${codes_file} --vocabulary "${vocab_file}.en" --vocabulary-threshold 2 < ${dev_src} > ${dev_src_out}
subword-nmt apply-bpe -c ${codes_file} --vocabulary "${vocab_file}.${lang}" --vocabulary-threshold 2 < ${dev_tgt} > ${dev_tgt_out}

subword-nmt apply-bpe -c ${codes_file} --vocabulary "${vocab_file}.en" --vocabulary-threshold 2 < ${test_src} > ${test_src_out}
subword-nmt apply-bpe -c ${codes_file} --vocabulary "${vocab_file}.${lang}" --vocabulary-threshold 2 < ${test_tgt} > ${test_tgt_out}

cp ${test_src} ${write_dir}/test.en.ori
cp ${test_tgt} ${write_dir}/test.${lang}.ori
cp ${dev_src} ${write_dir}/dev.en.ori
cp ${dev_tgt} ${write_dir}/dev.${lang}.ori
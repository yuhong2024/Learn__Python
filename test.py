def extract_fasta_headers(input_file, output_file):
    """
    从FASTA文件中提取标题并保存到txt文件

    参数:
        input_file: 输入的FASTA文件路径
        output_file: 输出的txt文件路径
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.startswith('>'):
                # 去除'>'并去除两端空白字符后写入输出文件
                header = line[1:].strip()
                outfile.write(header + '\n')


# 使用示例
input_path = r"D:\run\0330\napus\ap2.fas"
output_path = r"D:\run\0330\napus\headers.txt"  # 你可以修改输出路径

extract_fasta_headers(input_path, output_path)
print(f"标题已提取并保存到 {output_path}")
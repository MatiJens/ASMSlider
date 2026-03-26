from asmslider import ASMSlider


input_fasta = "/mnt/lustre/data/other/fusarium.fa"
output_dir = "/mnt/magisterka/results/slider/fusarium"

ASMSlider.scan(
    input_fasta=input_fasta,
    output_dir=output_dir,
    prefix="fusarium",
    threshold=0.05,
    merge_distance=3,
    stride=2,
    batch_size=1024,
)

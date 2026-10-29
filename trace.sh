# docker run -it --rm \
#     --ipc=host \
#     --privileged \
#     --cap-add=CAP_SYS_ADMIN \
#     --device=/dev/kfd \
#     --device=/dev/dri \
#     --device=/dev/mem \
#     --group-add render \
#     --cap-add=SYS_PTRACE \
#     --security-opt seccomp=unconfined \
#     -v $(pwd):$(pwd) \
#     -w $(pwd) \
#     --name nguyentran-vllm \
#     --entrypoint bash \
#     vllm/vllm-openai-rocm:v0.23.0

rocprofv3 --att --att-activity 10 -d profile_gemm -- ./run 6000

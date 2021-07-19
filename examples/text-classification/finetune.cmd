executable = finetune.sh
getenv     = true
output     = finetune_glt2.out
error      = finetune_glt2.error
log        = finetune_glt2.log
notification = complete
arguments = ""
transfer_executable = false
request_memory = 5*1024
request_GPUs = 1
Requirements = ( ( machine == "patas-gn2.ling.washington.edu" ) )
queue

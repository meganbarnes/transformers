executable = finetune.sh
getenv     = true
output     = finetune_gn3.out
error      = finetune_gn3.error
log        = finetune_gn3.log
notification = complete
arguments = ""
transfer_executable = false
request_memory = 5*1024
request_GPUs = 1
Requirements = ( ( machine == "patas-gn3.ling.washington.edu" ) )
queue

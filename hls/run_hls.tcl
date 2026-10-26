set PROJ    lr_consensus_proj
set SOLN    solution1
set PART    {xc7z020clg400-1}
set PERIOD  10
set RUN_COSIM  1
set RUN_EXPORT 1

open_project -reset $PROJ
set_top lr_consensus

add_files     [file normalize lr_consensus.cpp]
add_files -tb [file normalize lr_tb.cpp]       
add_files -tb [file normalize test_vectors.txt]

open_solution -reset -flow_target vivado $SOLN
set_part $PART
create_clock -period $PERIOD -name default

puts "\n=== C SIMULATION ==="
csim_design

puts "\n=== C SYNTHESIS ==="
csynth_design
puts "report: $PROJ/$SOLN/syn/report/lr_consensus_csynth.rpt"

if {$RUN_COSIM} {
    puts "\n=== C/RTL CO-SIMULATION ==="
    cosim_design -rtl verilog -trace_level all
    puts "waveform: $PROJ/$SOLN/sim/verilog/"
}

if {$RUN_EXPORT} {
    puts "\n=== IP EXPORT ==="
    export_design -format ip_catalog -rtl verilog \
        -display_name "Relay-Local LR Consensus Detector" \
        -vendor "gdidomenico" -library "grid" -version "1.0"
    puts "IP: $PROJ/$SOLN/impl/ip/"
}

puts "\n=== DONE ==="
exit

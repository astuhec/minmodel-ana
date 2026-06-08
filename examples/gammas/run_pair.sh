#!/bin/bash
Delta=$1
gap=$2
LOGDIR="logs"
mkdir -p "$LOGDIR"
python3 run_gapgamma.py "$Delta" "$gap" > "$LOGDIR/run_${Delta}_${gap}.log" 2>&1
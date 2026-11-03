#!/usr/bin/env bash
# round.sh <n> <stockPair> : stock on <stockPair>, overlay on the other pair, concurrently
cd "$(dirname "$0")"; n=$1 s=$2; o=$([[ $s == A ]] && echo B || echo A)
./arm.sh stock $s $n > r$n-stock.console 2>&1 & ./arm.sh overlay $o $n > r$n-overlay.console 2>&1 & wait

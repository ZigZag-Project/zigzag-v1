# Console information

To run a complete DSE for a certain workload, the tool will go through
several steps. The information printed on console shows which step is
now processing and some results for already finished steps.

Some **abbreviations** are used to keep the printing clean and concise:


```
MSG - Memory Scheme Generator
SUG - Spatial Unrolling Generator
TMG - Temporal Mapping Generator
CM - Cost Model

L - neural network Layer
M - Memory scheme
SU - Spatial Unrolling
TM - Temporal Mapping

en - energy
ut - mac array utilization
```

## Example
```
ZigZag started running.
```
```
10:16:08  MSG started
 Area-fitting memory combination:  135 / 135 | Valid hierarchy found:  8
10:16:08  MSG finished
```

The basic information of all the valid memory schemes found by MSG will
be listed, in which you see 1) **memory size** for each operand (Weight,
Input, Output) at each memory level, starting from the innermost memory
level, and 2) **memory unrolling**, indicating how many memory modules
at each level are deployed.
```
MEM HIERARCHY  1 / 8
{'W': [512.0, 1048576.0, 16777216], 'I': [512.0, 1048576.0, 16777216], 'O': [512.0, 1048576.0, 16777216]}
{'W': [168, 1, 1], 'I': [168, 1, 1], 'O': [168, 1, 1]}

MEM HIERARCHY  2 / 8
{'W': [512.0, 16777216], 'I': [512.0, 524288.0, 16777216], 'O': [512.0, 524288.0, 16777216]}
{'W': [168, 1], 'I': [168, 1, 1], 'O': [168, 1, 1]}

MEM HIERARCHY  3 / 8
{'W': [512.0, 524288.0, 16777216], 'I': [512.0, 524288.0, 16777216], 'O': [512.0, 16777216]}
{'W': [168, 1, 1], 'I': [168, 1, 1], 'O': [168, 1]}
...
```

For each memory scheme found, SUG will search for valid spatial
unrolling.

```
10:16:09  L 4 , M 1 / 8  SUG started
10:16:09  L 4 , M 1 / 8  SUG finished | Valid SU found 15
...
10:16:09  L 4 , M 5 / 8  SUG started
10:16:09  L 4 , M 5 / 8  SUG finished | Valid SU found 15
...
```

For each spatial unrolling found, TMG will search for valid temporal mapping.

If the ```temporal_mapping_search_method``` is set as ```iterative``` in
the setting file, the tool will print both number of partial and
final TMs found. Otherwise it will only print the number of total valid TMs found.

```
10:16:13  L 4 , M 1 / 8 , SU 1 / 15  TMG started
10:19:02  L 4 , M 1 / 8 , SU 1 / 15  TMG finished | Valid TM found ( partial: 1376 , final: 138 )
...
10:16:14  L 4 , M 5 / 8 , SU 14 / 15  TMG started
10:17:57  L 4 , M 5 / 8 , SU 14 / 15  TMG finished | Valid TM found ( partial: 1697 , final: 386 )
...
```

After TMG finishes, all the valid TMs will be fed into CM to
evaluate the HW cost, both energy and performance. We use MAC array
utilization as the metrics for performance measurement.

After CM finishes, the tool will print the basic information of the two
optimum design points it found: 1) the TM that minimizes the energy
('min en') and 2) the TM that maximizes the MAC array utilization ('max
ut'), same as maximizing the throughput. Note that sometimes these two
design points can be the overlapped.

A concise hardware cost summary for each design points is given in
tuple format: **(energy, MAC array utilization, area)**.
```
10:19:02  L 4 , M 1 / 8 , SU 1 / 15  CM  started
10:19:03  L 4 , M 1 / 8 , SU 1 / 15  CM  finished | Elapsed time: 170 sec | [min en: (2413267813, 0.08, 2642635) max ut: (2492528361, 0.08, 2642635)] in all TMs
...
10:17:57  L 4 , M 5 / 8 , SU 14 / 15  CM  started
10:18:05  L 4 , M 5 / 8 , SU 14 / 15  CM  finished | Elapsed time: 111 sec | [min en: (2296576246, 0.62, 2711284) max ut: (2299454127, 0.83, 2711284)] in all TMs
...
```

For every memory scheme, after all the SUs of that memory scheme have
been evaluated, the best ones are printed.

The example below shows that for memory scheme 1 and 5, their best SU
indexes are 15 & 8 and 11 & 11 respectively.

```
10:19:11  L 4,  M 1,  SU 15  Min En: (2250475032, 0.13, 2642635) in all SUs and TMs
10:19:11  L 4,  M 1,  SU 8  Max Ut: (2340634156, 0.22, 2642635) in all SUs and TMs
...
10:19:40  L 4,  M 5,  SU 11  Min En: (2249748101, 0.83, 2711284) in all SUs and TMs
10:19:40  L 4,  M 5,  SU 11  Max Ut: (2249748101, 0.83, 2711284) in all SUs and TMs
...
```

Finally, after all the memory schemes have been evaluated, the best ones are printed.

The example below shows that to run layer 4, memory scheme 2 with SU 15
achieves the global minimal energy; memory scheme 7 with SU 13 achieves
the global maximum MAC array utilization.

```
10:27:26  L 4,  M 2,  SU 15  Min En: (2206348947, 0.43, 2690648) in all MEMs, SUs, and TMs
10:27:26  L 4,  M 7,  SU 13  Max Ut: (2210541731, 0.83, 2711284) in all MEMs, SUs, and TMs
```

```
ZigZag finished running. Total elapsed time: 678 seconds.
Results are saved to ./user_defined_path.
```

### Comments
1. This is an example of running the complete DSE. If user runs the tool
   in other modes, like only search for the best TM with fixed
   architecture and SU, the information printed on console will be a
   subset of the example above.
2. If user applies the multiprocessing attribute of the tool, the
   printed results will not follow the index order.

# Example result files

The example result files and their documentation files are given under this folder with ```concise version``` and ```complete version```.
We recommend user start understanding output data format from the ```concise version``` by going through [concise output file](./example_result_file_concise_version.pdf) and [concise documentation](./result_file_documentation_concise_version.pdf) side by side. Then, more interesting and insightful informaiton to be explored in [complete output file](./example_result_file_complete_version.pdf) and [complete documentation](./result_file_documentation_complete_version.pdf).

Each **result file (.xml)** goes with a corresponding **mapping file
(.mapping)** to help user better visualize and understand the mapping
scheme of the design point. For example:

``` python
*************************** Levels In The System ***************************

 W: ['MAC', 'rf512', 'sram128kb', 'sram16Mb']                              
 I: ['MAC', 'rf512', 'sram16kb', 'sram16Mb']                               
 O: ['MAC', 'rf512', 'sram128kb', 'sram16Mb']  
 
 
 ********************* Spatial Unrolling Visualization *********************
 
 W: [[], [[('OY', 13)], [('FY', 3), ('C', 4)]], [], []]                    
 I: [[], [[('OY', 13)], [('FY', 3), ('C', 4)]], [], []]                    
 O: [[], [[('OY', 13)], [('FY', 3), ('C', 4)]], [], []]                    
                                                 
---------------------------------------------------------------------------
 Unrolled Loops           W                 I                 O            
---------------------------------------------------------------------------
 unroll C in [0:4)        rf512 (D2)        rf512 (D2)        rf512 (D2)   
---------------------------------------------------------------------------
 unroll FY in [0:3)       rf512 (D2)        rf512 (D2)        rf512 (D2)   
---------------------------------------------------------------------------
 unroll OY in [0:13)      rf512 (D1)        rf512 (D1)        rf512 (D1)   
---------------------------------------------------------------------------
 (Notes: Unrolled loops' order doesn't matters; D1 and D2 indicate 2D PE array's two dimensions.)


**************************** Temporal Mapping Visualization **************************** 

 W: [[('C', 8), ('FX', 3)], [('K', 48), ('C', 2), ('OX', 13)], [('K', 4), ('C', 3), ('K', 2)]]
 I: [[('C', 8), ('FX', 3), ('K', 48)], [('C', 2)], [('OX', 13), ('K', 4), ('C', 3), ('K', 2)]]
 O: [[('C', 8), ('FX', 3)], [('K', 48), ('C', 2), ('OX', 13), ('K', 4), ('C', 3)], [('K', 2)]]
                                                                                         
-----------------------------------------------------------------------------------------
 Temporal Loops                      W                  I                  O             
-----------------------------------------------------------------------------------------
 for K in [0:2)                      sram16Mb           sram16Mb           sram16Mb      
-----------------------------------------------------------------------------------------
  for C in [0:3)                     sram16Mb           sram16Mb           sram128kb     
-----------------------------------------------------------------------------------------
   for K in [0:4)                    sram16Mb           sram16Mb           sram128kb     
-----------------------------------------------------------------------------------------
    for OX in [0:13)                 sram128kb          sram16Mb           sram128kb     
-----------------------------------------------------------------------------------------
     for C in [0:2)                  sram128kb          sram16kb           sram128kb     
-----------------------------------------------------------------------------------------
      for K in [0:48)                sram128kb          rf512              sram128kb     
-----------------------------------------------------------------------------------------
       for FX in [0:3)               rf512              rf512              rf512         
-----------------------------------------------------------------------------------------
        for C in [0:8)               rf512              rf512              rf512         
-----------------------------------------------------------------------------------------
 (Notes: Temporal Mapping starts from the innermost memory level. MAC level is out of Temporal Mapping's scope.)
```

(User can open .xml file with Web browser APPs, like Firefox, Chrome,
IE, etc., to view it nicely.)

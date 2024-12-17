# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/alexbovet/flow_stability/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                               |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/flowstab/FlowStability.py                      |      976 |      883 |     10% |61-62, 88-115, 119, 125-132, 139-143, 147-150, 162-186, 190-191, 195, 201-214, 278-365, 378-386, 398-424, 437-439, 448, 467-500, 518-556, 562, 580-582, 593-695, 709-748, 754-768, 787-876, 880-891, 895, 900, 904, 908, 912-913, 917-920, 925-927, 932, 987-998, 1009-1012, 1023-1040, 1052-1069, 1083-1102, 1113-1215, 1223-1229, 1237, 1243, 1256-1289, 1292-1300, 1304, 1309, 1313, 1317, 1321, 1325, 1330, 1347-1361, 1417-1501, 1512, 1525-1528, 1539-1556, 1568-1579, 1593-1604, 1620-1632, 1637, 1698-1822, 1828-1845, 1858-1956, 2010-2038, 2049-2077, 2093-2149, 2163-2234, 2245-2265, 2308-2385, 2392, 2421-2442, 2449-2482 |
| src/flowstab/SparseStochMat.py                     |      405 |      209 |     48% |47-48, 56-57, 179, 190, 226-232, 239-248, 260, 268, 274, 285, 294, 302-306, 312, 316-321, 326, 334-362, 369, 374-401, 405-408, 416-504, 511-514, 519-528, 534, 559-588, 641, 672, 727, 859-873, 877-887, 904-916, 933-946, 968-1009, 1018-1031, 1042-1055, 1076-1089, 1122-1150, 1159-1162, 1172-1177 |
| src/flowstab/SynthTempNetwork.py                   |      276 |      117 |     58% |133, 153, 159, 164, 183, 188, 203, 208, 278, 285-287, 299, 327, 331, 337, 341-345, 378, 402, 426, 433-509, 584-586, 597, 638-640, 647-731 |
| src/flowstab/TemporalNetwork.py                    |     1029 |      888 |     14% |119, 129-134, 188, 230-273, 317-363, 416-525, 538-651, 662-747, 761-831, 845-918, 928-1002, 1084-1096, 1136-1286, 1344-1407, 1431-1519, 1533-1616, 1626-1698, 1705-1739, 1774-1784, 1786, 1822-1871, 1922-1943, 1962-2070, 2094, 2127, 2171-2191, 2209-2245, 2272-2334, 2343-2350, 2354-2376, 2391-2396, 2412-2465, 2504-2543, 2576-2592, 2609-2613, 2621, 2629-2646 |
| src/flowstab/\_cython\_sparse\_stoch\_subst.py     |       23 |        4 |     83% |     37-40 |
| src/flowstab/\_cython\_subst.py                    |      263 |       78 |     70% |75, 164, 167, 267, 270, 320-327, 374, 388, 405-430, 449-468, 485-511, 531-541, 558-571 |
| src/flowstab/parallel\_clustering.py               |       93 |       76 |     18% |45-101, 106, 114-118, 124-140, 146-170, 174-198, 204-229, 233-246 |
| src/flowstab/parallel\_expm.py                     |      109 |       94 |     14% |39-43, 47-60, 92-122, 127-143, 147, 179-268 |
| src/flowstab/scripts/run\_clusterings.py           |      462 |      462 |      0% |    42-994 |
| src/flowstab/scripts/run\_cov\_integrals.py        |      339 |      339 |      0% |    41-740 |
| src/flowstab/scripts/run\_laplacians\_transmats.py |      253 |      253 |      0% |    31-571 |
|                                          **TOTAL** | **4228** | **3403** | **20%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/alexbovet/flow_stability/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/alexbovet/flow_stability/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/alexbovet/flow_stability/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/alexbovet/flow_stability/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Falexbovet%2Fflow_stability%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/alexbovet/flow_stability/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.
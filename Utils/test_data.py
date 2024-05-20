# filtering

test1_leaves = [
    {
        "paRoute": {"smiles": "CC(C)(C)[C@H](N)CO", "type": "mol", "in_stock": True},
        "filtered": {"smiles": "CC(C)(C)[C@H](N)CO"},
    },
    {
        "paRoute": {"smiles": "Fc1cnc(Cl)nc1Cl", "type": "mol", "in_stock": True},
        "filtered": {"smiles": "Fc1cnc(Cl)nc1Cl"},
    },
]

test2_depth1 = [
    {
        "paRoute": {
            "smiles": "CC(C)(C)[C@@H](CO)Nc1nc(Cl)ncc1F",
            "type": "mol",
            "in_stock": False,
            "children": [
                {
                    "type": "reaction",
                    "smiles": "",
                    "metadata": {
                        "smiles": "Cl[c:9]1[n:10][c:11]([Cl:12])[n:13][cH:14][c:15]1[F:16].[CH3:1][C:2]([CH3:3])([CH3:4])[C@@H:5]([CH2:6][OH:7])[NH2:8]>>[CH3:1][C:2]([CH3:3])([CH3:4])[C@@H:5]([CH2:6][OH:7])[NH:8][c:9]1[n:10][c:11]([Cl:12])[n:13][cH:14][c:15]1[F:16]",
                        "rsmi": "Cl[c:9]1[n:10][c:11]([Cl:12])[n:13][cH:14][c:15]1[F:16].[CH3:1][C:2]([CH3:3])([CH3:4])[C@@H:5]([CH2:6][OH:7])[NH2:8]>CCN(CC)CC.CN(C)C=O.Cl.N>[CH3:1][C:2]([CH3:3])([CH3:4])[C@@H:5]([CH2:6][OH:7])[NH:8][c:9]1[n:10][c:11]([Cl:12])[n:13][cH:14][c:15]1[F:16]",
                        "reaction_hash": "JBULSURVMXPBNA-RXMQYKEDSA-N.WHPFEQUEHBULBW-UHFFFAOYSA-N>>YUFRLVWCUYFCMH-SSDOTTSWSA-N",
                        "ID": "US20140094473A1;0337;1510270",
                        "RingBreaker": False,
                    },
                    "children": [
                        {
                            "smiles": "CC(C)(C)[C@H](N)CO",
                            "type": "mol",
                            "in_stock": True,
                        },
                        {
                            "smiles": "Fc1cnc(Cl)nc1Cl",
                            "type": "mol",
                            "in_stock": True,
                        },
                    ],
                }
            ],
        },
        "filtered": {
            "smiles": "CC(C)(C)[C@@H](CO)Nc1nc(Cl)ncc1F",
            "children": [
                {"smiles": "CC(C)(C)[C@H](N)CO"},
                {"smiles": "Fc1cnc(Cl)nc1Cl"},
            ],
        },
    },
]

test3_depth2 = [
    {
        "paRoute": {
            "smiles": "CC(C)(C)[C@@H](CS(C)(=O)=O)Nc1nc(-c2c[nH]c3ncc(F)cc23)ncc1F",
            "type": "mol",
            "in_stock": False,
            "children": [
                {
                    "type": "reaction",
                    "smiles": "",
                    "metadata": {
                        "smiles": "Cc1ccc(S(=O)(=O)[n:17]2[cH:16][c:15](-[c:14]3[n:13][c:12]([NH:11][C@@H:5]([C:2]([CH3:1])([CH3:3])[CH3:4])[CH2:6][S:7]([CH3:8])(=[O:9])=[O:10])[c:27]([F:28])[cH:26][n:25]3)[c:24]3[c:18]2[n:19][cH:20][c:21]([F:22])[cH:23]3)cc1>>[CH3:1][C:2]([CH3:3])([CH3:4])[C@@H:5]([CH2:6][S:7]([CH3:8])(=[O:9])=[O:10])[NH:11][c:12]1[n:13][c:14](-[c:15]2[cH:16][nH:17][c:18]3[n:19][cH:20][c:21]([F:22])[cH:23][c:24]23)[n:25][cH:26][c:27]1[F:28]",
                        "rsmi": "Cc1ccc(S(=O)(=O)[n:17]2[cH:16][c:15](-[c:14]3[n:13][c:12]([NH:11][C@@H:5]([C:2]([CH3:1])([CH3:3])[CH3:4])[CH2:6][S:7]([CH3:8])(=[O:9])=[O:10])[c:27]([F:28])[cH:26][n:25]3)[c:24]3[c:18]2[n:19][cH:20][c:21]([F:22])[cH:23]3)cc1>C1CCOC1.CO.Cl.N.[Na+]>[CH3:1][C:2]([CH3:3])([CH3:4])[C@@H:5]([CH2:6][S:7]([CH3:8])(=[O:9])=[O:10])[NH:11][c:12]1[n:13][c:14](-[c:15]2[cH:16][nH:17][c:18]3[n:19][cH:20][c:21]([F:22])[cH:23][c:24]23)[n:25][cH:26][c:27]1[F:28]",
                        "reaction_hash": "BYRRRVGVJQXRHJ-OAQYLSRUSA-N>>IVAKVHLWFYMRRP-CQSZACIVSA-N",
                        "ID": "US20140094473A1;0415;1510303",
                        "RingBreaker": False,
                    },
                    "children": [
                        {
                            "smiles": "Cc1ccc(S(=O)(=O)n2cc(-c3ncc(F)c(N[C@H](CS(C)(=O)=O)C(C)(C)C)n3)c3cc(F)cnc32)cc1",
                            "type": "mol",
                            "in_stock": False,
                            "children": [
                                {
                                    "type": "reaction",
                                    "smiles": "",
                                    "metadata": {
                                        "smiles": "CC1(C)OB([c:11]2[cH:10][n:9]([S:6]([c:5]3[cH:4][cH:3][c:2]([CH3:1])[cH:38][cH:37]3)(=[O:7])=[O:8])[c:36]3[c:30]2[cH:31][c:32]([F:33])[cH:34][n:35]3)OC1(C)C.Cl[c:12]1[n:13][cH:14][c:15]([F:16])[c:17]([NH:18][C@H:19]([CH2:20][S:21]([CH3:22])(=[O:23])=[O:24])[C:25]([CH3:26])([CH3:27])[CH3:28])[n:29]1>>[CH3:1][c:2]1[cH:3][cH:4][c:5]([S:6](=[O:7])(=[O:8])[n:9]2[cH:10][c:11](-[c:12]3[n:13][cH:14][c:15]([F:16])[c:17]([NH:18][C@H:19]([CH2:20][S:21]([CH3:22])(=[O:23])=[O:24])[C:25]([CH3:26])([CH3:27])[CH3:28])[n:29]3)[c:30]3[cH:31][c:32]([F:33])[cH:34][n:35][c:36]23)[cH:37][cH:38]1",
                                        "rsmi": "CC1(C)OB([c:11]2[cH:10][n:9]([S:6]([c:5]3[cH:4][cH:3][c:2]([CH3:1])[cH:38][cH:37]3)(=[O:7])=[O:8])[c:36]3[c:30]2[cH:31][c:32]([F:33])[cH:34][n:35]3)OC1(C)C.Cl[c:12]1[n:13][cH:14][c:15]([F:16])[c:17]([NH:18][C@H:19]([CH2:20][S:21]([CH3:22])(=[O:23])=[O:24])[C:25]([CH3:26])([CH3:27])[CH3:28])[n:29]1>CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1.O.O=C(/C=C/c1ccccc1)/C=C/c1ccccc1.O=C(/C=C/c1ccccc1)/C=C/c1ccccc1.O=C(/C=C/c1ccccc1)/C=C/c1ccccc1.O=P(O)(O)O.[K+].[K+].[K+].[Pd].[Pd]>[CH3:1][c:2]1[cH:3][cH:4][c:5]([S:6](=[O:7])(=[O:8])[n:9]2[cH:10][c:11](-[c:12]3[n:13][cH:14][c:15]([F:16])[c:17]([NH:18][C@H:19]([CH2:20][S:21]([CH3:22])(=[O:23])=[O:24])[C:25]([CH3:26])([CH3:27])[CH3:28])[n:29]3)[c:30]3[cH:31][c:32]([F:33])[cH:34][n:35][c:36]23)[cH:37][cH:38]1",
                                        "reaction_hash": "LLXFXTYIUJKMBA-MRVPVSSYSA-N.NBTQILYOTKPKRZ-UHFFFAOYSA-N>>BYRRRVGVJQXRHJ-OAQYLSRUSA-N",
                                        "ID": "US20140094473A1;0414;1510302",
                                        "RingBreaker": False,
                                    },
                                    "children": [
                                        {
                                            "smiles": "CC(C)(C)[C@@H](CS(C)(=O)=O)Nc1nc(Cl)ncc1F",
                                            "type": "mol",
                                            "in_stock": False,
                                        },
                                        {
                                            "smiles": "Cc1ccc(S(=O)(=O)n2cc(B3OC(C)(C)C(C)(C)O3)c3cc(F)cnc32)cc1",
                                            "type": "mol",
                                            "in_stock": True,
                                        },
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        },
        "filtered": {
            "smiles": "CC(C)(C)[C@@H](CS(C)(=O)=O)Nc1nc(-c2c[nH]c3ncc(F)cc23)ncc1F",
            "children": [
                {
                    "smiles": "Cc1ccc(S(=O)(=O)n2cc(-c3ncc(F)c(N[C@H](CS(C)(=O)=O)C(C)(C)C)n3)c3cc(F)cnc32)cc1",
                    "children": [
                        {"smiles": "CC(C)(C)[C@@H](CS(C)(=O)=O)Nc1nc(Cl)ncc1F"},
                        {
                            "smiles": "Cc1ccc(S(=O)(=O)n2cc(B3OC(C)(C)C(C)(C)O3)c3cc(F)cnc32)cc1"
                        },
                    ],
                }
            ],
        },
    }
]


n1_route_idx0_paRoute = {
    "smiles": "CC(C)(C)[C@@H](CS(C)(=O)=O)Nc1nc(-c2c[nH]c3ncc(F)cc23)ncc1F",
    "type": "mol",
    "in_stock": False,
    "children": [
        {
            "type": "reaction",
            "smiles": "",
            "metadata": {
                "smiles": "Cc1ccc(S(=O)(=O)[n:17]2[cH:16][c:15](-[c:14]3[n:13][c:12]([NH:11][C@@H:5]([C:2]([CH3:1])([CH3:3])[CH3:4])[CH2:6][S:7]([CH3:8])(=[O:9])=[O:10])[c:27]([F:28])[cH:26][n:25]3)[c:24]3[c:18]2[n:19][cH:20][c:21]([F:22])[cH:23]3)cc1>>[CH3:1][C:2]([CH3:3])([CH3:4])[C@@H:5]([CH2:6][S:7]([CH3:8])(=[O:9])=[O:10])[NH:11][c:12]1[n:13][c:14](-[c:15]2[cH:16][nH:17][c:18]3[n:19][cH:20][c:21]([F:22])[cH:23][c:24]23)[n:25][cH:26][c:27]1[F:28]",
                "rsmi": "Cc1ccc(S(=O)(=O)[n:17]2[cH:16][c:15](-[c:14]3[n:13][c:12]([NH:11][C@@H:5]([C:2]([CH3:1])([CH3:3])[CH3:4])[CH2:6][S:7]([CH3:8])(=[O:9])=[O:10])[c:27]([F:28])[cH:26][n:25]3)[c:24]3[c:18]2[n:19][cH:20][c:21]([F:22])[cH:23]3)cc1>C1CCOC1.CO.Cl.N.[Na+]>[CH3:1][C:2]([CH3:3])([CH3:4])[C@@H:5]([CH2:6][S:7]([CH3:8])(=[O:9])=[O:10])[NH:11][c:12]1[n:13][c:14](-[c:15]2[cH:16][nH:17][c:18]3[n:19][cH:20][c:21]([F:22])[cH:23][c:24]23)[n:25][cH:26][c:27]1[F:28]",
                "reaction_hash": "BYRRRVGVJQXRHJ-OAQYLSRUSA-N>>IVAKVHLWFYMRRP-CQSZACIVSA-N",
                "ID": "US20140094473A1;0415;1510303",
                "RingBreaker": False,
            },
            "children": [
                {
                    "smiles": "Cc1ccc(S(=O)(=O)n2cc(-c3ncc(F)c(N[C@H](CS(C)(=O)=O)C(C)(C)C)n3)c3cc(F)cnc32)cc1",
                    "type": "mol",
                    "in_stock": False,
                    "children": [
                        {
                            "type": "reaction",
                            "smiles": "",
                            "metadata": {
                                "smiles": "CC1(C)OB([c:11]2[cH:10][n:9]([S:6]([c:5]3[cH:4][cH:3][c:2]([CH3:1])[cH:38][cH:37]3)(=[O:7])=[O:8])[c:36]3[c:30]2[cH:31][c:32]([F:33])[cH:34][n:35]3)OC1(C)C.Cl[c:12]1[n:13][cH:14][c:15]([F:16])[c:17]([NH:18][C@H:19]([CH2:20][S:21]([CH3:22])(=[O:23])=[O:24])[C:25]([CH3:26])([CH3:27])[CH3:28])[n:29]1>>[CH3:1][c:2]1[cH:3][cH:4][c:5]([S:6](=[O:7])(=[O:8])[n:9]2[cH:10][c:11](-[c:12]3[n:13][cH:14][c:15]([F:16])[c:17]([NH:18][C@H:19]([CH2:20][S:21]([CH3:22])(=[O:23])=[O:24])[C:25]([CH3:26])([CH3:27])[CH3:28])[n:29]3)[c:30]3[cH:31][c:32]([F:33])[cH:34][n:35][c:36]23)[cH:37][cH:38]1",
                                "rsmi": "CC1(C)OB([c:11]2[cH:10][n:9]([S:6]([c:5]3[cH:4][cH:3][c:2]([CH3:1])[cH:38][cH:37]3)(=[O:7])=[O:8])[c:36]3[c:30]2[cH:31][c:32]([F:33])[cH:34][n:35]3)OC1(C)C.Cl[c:12]1[n:13][cH:14][c:15]([F:16])[c:17]([NH:18][C@H:19]([CH2:20][S:21]([CH3:22])(=[O:23])=[O:24])[C:25]([CH3:26])([CH3:27])[CH3:28])[n:29]1>CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1.O.O=C(/C=C/c1ccccc1)/C=C/c1ccccc1.O=C(/C=C/c1ccccc1)/C=C/c1ccccc1.O=C(/C=C/c1ccccc1)/C=C/c1ccccc1.O=P(O)(O)O.[K+].[K+].[K+].[Pd].[Pd]>[CH3:1][c:2]1[cH:3][cH:4][c:5]([S:6](=[O:7])(=[O:8])[n:9]2[cH:10][c:11](-[c:12]3[n:13][cH:14][c:15]([F:16])[c:17]([NH:18][C@H:19]([CH2:20][S:21]([CH3:22])(=[O:23])=[O:24])[C:25]([CH3:26])([CH3:27])[CH3:28])[n:29]3)[c:30]3[cH:31][c:32]([F:33])[cH:34][n:35][c:36]23)[cH:37][cH:38]1",
                                "reaction_hash": "LLXFXTYIUJKMBA-MRVPVSSYSA-N.NBTQILYOTKPKRZ-UHFFFAOYSA-N>>BYRRRVGVJQXRHJ-OAQYLSRUSA-N",
                                "ID": "US20140094473A1;0414;1510302",
                                "RingBreaker": False,
                            },
                            "children": [
                                {
                                    "smiles": "CC(C)(C)[C@@H](CS(C)(=O)=O)Nc1nc(Cl)ncc1F",
                                    "type": "mol",
                                    "in_stock": False,
                                    "children": [
                                        {
                                            "type": "reaction",
                                            "smiles": "",
                                            "metadata": {
                                                "smiles": "[CH3:1][C:2]([CH3:3])([CH3:4])[C@@H:5]([CH2:6][S:7][CH3:8])[NH:11][c:12]1[n:13][c:14]([Cl:15])[n:16][cH:17][c:18]1[F:19].OS(O[OH:10])=[O:9]>>[CH3:1][C:2]([CH3:3])([CH3:4])[C@@H:5]([CH2:6][S:7]([CH3:8])(=[O:9])=[O:10])[NH:11][c:12]1[n:13][c:14]([Cl:15])[n:16][cH:17][c:18]1[F:19]",
                                                "rsmi": "[CH3:1][C:2]([CH3:3])([CH3:4])[C@@H:5]([CH2:6][S:7][CH3:8])[NH:11][c:12]1[n:13][c:14]([Cl:15])[n:16][cH:17][c:18]1[F:19].OS(O[OH:10])=[O:9]>CO.O.[K+]>[CH3:1][C:2]([CH3:3])([CH3:4])[C@@H:5]([CH2:6][S:7]([CH3:8])(=[O:9])=[O:10])[NH:11][c:12]1[n:13][c:14]([Cl:15])[n:16][cH:17][c:18]1[F:19]",
                                                "reaction_hash": "AYSIGUCDFZQVQT-UHFFFAOYSA-N.SPMAUDSBFHMZDG-MRVPVSSYSA-N>>LLXFXTYIUJKMBA-MRVPVSSYSA-N",
                                                "ID": "US20140094473A1;0413;1510301",
                                                "RingBreaker": False,
                                            },
                                            "children": [
                                                {
                                                    "smiles": "CSC[C@@H](Nc1nc(Cl)ncc1F)C(C)(C)C",
                                                    "type": "mol",
                                                    "in_stock": False,
                                                    "children": [
                                                        {
                                                            "type": "reaction",
                                                            "smiles": "",
                                                            "metadata": {
                                                                "smiles": "I[CH3:1].[SH:2][CH2:3][C@@H:4]([NH:5][c:6]1[n:7][c:8]([Cl:9])[n:10][cH:11][c:12]1[F:13])[C:14]([CH3:15])([CH3:16])[CH3:17]>>[CH3:1][S:2][CH2:3][C@@H:4]([NH:5][c:6]1[n:7][c:8]([Cl:9])[n:10][cH:11][c:12]1[F:13])[C:14]([CH3:15])([CH3:16])[CH3:17]",
                                                                "rsmi": "I[CH3:1].[SH:2][CH2:3][C@@H:4]([NH:5][c:6]1[n:7][c:8]([Cl:9])[n:10][cH:11][c:12]1[F:13])[C:14]([CH3:15])([CH3:16])[CH3:17]>CC(C)=O.O=C(O)O.[K+].[K+]>[CH3:1][S:2][CH2:3][C@@H:4]([NH:5][c:6]1[n:7][c:8]([Cl:9])[n:10][cH:11][c:12]1[F:13])[C:14]([CH3:15])([CH3:16])[CH3:17]",
                                                                "reaction_hash": "BOKGEBFFQFYPJZ-SSDOTTSWSA-N.INQOMBQAUSQDDS-UHFFFAOYSA-N>>SPMAUDSBFHMZDG-MRVPVSSYSA-N",
                                                                "ID": "US20140094473A1;0412;1510300",
                                                                "RingBreaker": False,
                                                            },
                                                            "children": [
                                                                {
                                                                    "smiles": "CC(C)(C)[C@@H](CS)Nc1nc(Cl)ncc1F",
                                                                    "type": "mol",
                                                                    "in_stock": False,
                                                                    "children": [
                                                                        {
                                                                            "type": "reaction",
                                                                            "smiles": "",
                                                                            "metadata": {
                                                                                "smiles": "CC(=O)[S:7][CH2:6][C@H:5]([C:2]([CH3:1])([CH3:3])[CH3:4])[NH:8][c:9]1[n:10][c:11]([Cl:12])[n:13][cH:14][c:15]1[F:16]>>[CH3:1][C:2]([CH3:3])([CH3:4])[C@@H:5]([CH2:6][SH:7])[NH:8][c:9]1[n:10][c:11]([Cl:12])[n:13][cH:14][c:15]1[F:16]",
                                                                                "rsmi": "CC(=O)[S:7][CH2:6][C@H:5]([C:2]([CH3:1])([CH3:3])[CH3:4])[NH:8][c:9]1[n:10][c:11]([Cl:12])[n:13][cH:14][c:15]1[F:16]>CO.CO.[Na+]>[CH3:1][C:2]([CH3:3])([CH3:4])[C@@H:5]([CH2:6][SH:7])[NH:8][c:9]1[n:10][c:11]([Cl:12])[n:13][cH:14][c:15]1[F:16]",
                                                                                "reaction_hash": "RMPGWJSJCMHPKS-SECBINFHSA-N>>BOKGEBFFQFYPJZ-SSDOTTSWSA-N",
                                                                                "ID": "US20140094473A1;0411;1510299",
                                                                                "RingBreaker": False,
                                                                            },
                                                                            "children": [
                                                                                {
                                                                                    "smiles": "CC(=O)SC[C@@H](Nc1nc(Cl)ncc1F)C(C)(C)C",
                                                                                    "type": "mol",
                                                                                    "in_stock": False,
                                                                                    "children": [
                                                                                        {
                                                                                            "type": "reaction",
                                                                                            "smiles": "",
                                                                                            "metadata": {
                                                                                                "smiles": "CS(=O)(=O)O[CH2:5][C@@H:6]([NH:7][c:8]1[n:9][c:10]([Cl:11])[n:12][cH:13][c:14]1[F:15])[C:16]([CH3:17])([CH3:18])[CH3:19].[CH3:1][C:2]([OH:3])=[S:4]>>[CH3:1][C:2](=[O:3])[S:4][CH2:5][C@@H:6]([NH:7][c:8]1[n:9][c:10]([Cl:11])[n:12][cH:13][c:14]1[F:15])[C:16]([CH3:17])([CH3:18])[CH3:19]",
                                                                                                "rsmi": "CS(=O)(=O)O[CH2:5][C@@H:6]([NH:7][c:8]1[n:9][c:10]([Cl:11])[n:12][cH:13][c:14]1[F:15])[C:16]([CH3:17])([CH3:18])[CH3:19].[CH3:1][C:2]([OH:3])=[S:4]>CN(C)C=O.O.[K+]>[CH3:1][C:2](=[O:3])[S:4][CH2:5][C@@H:6]([NH:7][c:8]1[n:9][c:10]([Cl:11])[n:12][cH:13][c:14]1[F:15])[C:16]([CH3:17])([CH3:18])[CH3:19]",
                                                                                                "reaction_hash": "BGUYNPATMDNWCR-MRVPVSSYSA-N.DUYAAUVXQSMXQP-UHFFFAOYSA-N>>RMPGWJSJCMHPKS-SECBINFHSA-N",
                                                                                                "ID": "US20140094473A1;0410;1510298",
                                                                                                "RingBreaker": False,
                                                                                            },
                                                                                            "children": [
                                                                                                {
                                                                                                    "smiles": "CC(C)(C)[C@@H](COS(C)(=O)=O)Nc1nc(Cl)ncc1F",
                                                                                                    "type": "mol",
                                                                                                    "in_stock": False,
                                                                                                    "children": [
                                                                                                        {
                                                                                                            "type": "reaction",
                                                                                                            "smiles": "",
                                                                                                            "metadata": {
                                                                                                                "smiles": "Cl[S:8]([CH3:9])(=[O:10])=[O:11].[CH3:1][C:2]([CH3:3])([CH3:4])[C@@H:5]([CH2:6][OH:7])[NH:12][c:13]1[n:14][c:15]([Cl:16])[n:17][cH:18][c:19]1[F:20]>>[CH3:1][C:2]([CH3:3])([CH3:4])[C@@H:5]([CH2:6][O:7][S:8]([CH3:9])(=[O:10])=[O:11])[NH:12][c:13]1[n:14][c:15]([Cl:16])[n:17][cH:18][c:19]1[F:20]",
                                                                                                                "rsmi": "Cl[S:8]([CH3:9])(=[O:10])=[O:11].[CH3:1][C:2]([CH3:3])([CH3:4])[C@@H:5]([CH2:6][OH:7])[NH:12][c:13]1[n:14][c:15]([Cl:16])[n:17][cH:18][c:19]1[F:20]>CCN(CC)CC.ClCCl>[CH3:1][C:2]([CH3:3])([CH3:4])[C@@H:5]([CH2:6][O:7][S:8]([CH3:9])(=[O:10])=[O:11])[NH:12][c:13]1[n:14][c:15]([Cl:16])[n:17][cH:18][c:19]1[F:20]",
                                                                                                                "reaction_hash": "QARBMVPHQWIHKH-UHFFFAOYSA-N.YUFRLVWCUYFCMH-SSDOTTSWSA-N>>BGUYNPATMDNWCR-MRVPVSSYSA-N",
                                                                                                                "ID": "US20140094473A1;0409;1510297",
                                                                                                                "RingBreaker": False,
                                                                                                            },
                                                                                                            "children": [
                                                                                                                {
                                                                                                                    "smiles": "CC(C)(C)[C@@H](CO)Nc1nc(Cl)ncc1F",
                                                                                                                    "type": "mol",
                                                                                                                    "in_stock": False,
                                                                                                                    "children": [
                                                                                                                        {
                                                                                                                            "type": "reaction",
                                                                                                                            "smiles": "",
                                                                                                                            "metadata": {
                                                                                                                                "smiles": "Cl[c:9]1[n:10][c:11]([Cl:12])[n:13][cH:14][c:15]1[F:16].[CH3:1][C:2]([CH3:3])([CH3:4])[C@@H:5]([CH2:6][OH:7])[NH2:8]>>[CH3:1][C:2]([CH3:3])([CH3:4])[C@@H:5]([CH2:6][OH:7])[NH:8][c:9]1[n:10][c:11]([Cl:12])[n:13][cH:14][c:15]1[F:16]",
                                                                                                                                "rsmi": "Cl[c:9]1[n:10][c:11]([Cl:12])[n:13][cH:14][c:15]1[F:16].[CH3:1][C:2]([CH3:3])([CH3:4])[C@@H:5]([CH2:6][OH:7])[NH2:8]>CCN(CC)CC.CN(C)C=O.Cl.N>[CH3:1][C:2]([CH3:3])([CH3:4])[C@@H:5]([CH2:6][OH:7])[NH:8][c:9]1[n:10][c:11]([Cl:12])[n:13][cH:14][c:15]1[F:16]",
                                                                                                                                "reaction_hash": "JBULSURVMXPBNA-RXMQYKEDSA-N.WHPFEQUEHBULBW-UHFFFAOYSA-N>>YUFRLVWCUYFCMH-SSDOTTSWSA-N",
                                                                                                                                "ID": "US20140094473A1;0337;1510270",
                                                                                                                                "RingBreaker": False,
                                                                                                                            },
                                                                                                                            "children": [
                                                                                                                                {
                                                                                                                                    "smiles": "CC(C)(C)[C@H](N)CO",
                                                                                                                                    "type": "mol",
                                                                                                                                    "in_stock": True,
                                                                                                                                },
                                                                                                                                {
                                                                                                                                    "smiles": "Fc1cnc(Cl)nc1Cl",
                                                                                                                                    "type": "mol",
                                                                                                                                    "in_stock": True,
                                                                                                                                },
                                                                                                                            ],
                                                                                                                        }
                                                                                                                    ],
                                                                                                                },
                                                                                                                {
                                                                                                                    "smiles": "CS(=O)(=O)Cl",
                                                                                                                    "type": "mol",
                                                                                                                    "in_stock": True,
                                                                                                                },
                                                                                                            ],
                                                                                                        }
                                                                                                    ],
                                                                                                },
                                                                                                {
                                                                                                    "smiles": "CC(O)=S",
                                                                                                    "type": "mol",
                                                                                                    "in_stock": True,
                                                                                                },
                                                                                            ],
                                                                                        }
                                                                                    ],
                                                                                }
                                                                            ],
                                                                        }
                                                                    ],
                                                                },
                                                                {
                                                                    "smiles": "CI",
                                                                    "type": "mol",
                                                                    "in_stock": True,
                                                                },
                                                            ],
                                                        }
                                                    ],
                                                },
                                                {
                                                    "smiles": "O=S(O)OO",
                                                    "type": "mol",
                                                    "in_stock": True,
                                                },
                                            ],
                                        }
                                    ],
                                },
                                {
                                    "smiles": "Cc1ccc(S(=O)(=O)n2cc(B3OC(C)(C)C(C)(C)O3)c3cc(F)cnc32)cc1",
                                    "type": "mol",
                                    "in_stock": True,
                                },
                            ],
                        }
                    ],
                }
            ],
        }
    ],
}

n1_route_idx0_out = {
    "smiles": "CC(C)(C)[C@@H](CS(C)(=O)=O)Nc1nc(-c2c[nH]c3ncc(F)cc23)ncc1F",
    "children": [
        {
            "smiles": "Cc1ccc(S(=O)(=O)n2cc(-c3ncc(F)c(N[C@H](CS(C)(=O)=O)C(C)(C)C)n3)c3cc(F)cnc32)cc1",
            "children": [
                {
                    "smiles": "CC(C)(C)[C@@H](CS(C)(=O)=O)Nc1nc(Cl)ncc1F",
                    "children": [
                        {
                            "smiles": "CSC[C@@H](Nc1nc(Cl)ncc1F)C(C)(C)C",
                            "children": [
                                {
                                    "smiles": "CC(C)(C)[C@@H](CS)Nc1nc(Cl)ncc1F",
                                    "children": [
                                        {
                                            "smiles": "CC(=O)SC[C@@H](Nc1nc(Cl)ncc1F)C(C)(C)C",
                                            "children": [
                                                {
                                                    "smiles": "CC(C)(C)[C@@H](COS(C)(=O)=O)Nc1nc(Cl)ncc1F",
                                                    "children": [
                                                        {
                                                            "smiles": "CC(C)(C)[C@@H](CO)Nc1nc(Cl)ncc1F",
                                                            "children": [
                                                                {
                                                                    "smiles": "CC(C)(C)[C@H](N)CO"
                                                                },
                                                                {
                                                                    "smiles": "Fc1cnc(Cl)nc1Cl"
                                                                },
                                                            ],
                                                        },
                                                        {"smiles": "CS(=O)(=O)Cl"},
                                                    ],
                                                },
                                                {"smiles": "CC(O)=S"},
                                            ],
                                        }
                                    ],
                                },
                                {"smiles": "CI"},
                            ],
                        },
                        {"smiles": "O=S(O)OO"},
                    ],
                },
                {"smiles": "Cc1ccc(S(=O)(=O)n2cc(B3OC(C)(C)C(C)(C)O3)c3cc(F)cnc32)cc1"},
            ],
        }
    ],
}

test4_n1route0 = [{"paRoute": n1_route_idx0_paRoute, "filtered": n1_route_idx0_out}]

test5_depth0_leaves = [
    {
        "filtered": test1_leaves[0]["filtered"],
        "leaves": ["CC(C)(C)[C@H](N)CO"],
    },
    {
        "filtered": test1_leaves[1]["filtered"],
        "leaves": ["Fc1cnc(Cl)nc1Cl"],
    },
]

test6_depth1_leaves = [
    {
        "filtered": test2_depth1[0]["filtered"],
        "leaves": ["CC(C)(C)[C@H](N)CO", "Fc1cnc(Cl)nc1Cl"],
    }
]

test7_depth2_leaves = [
    {
        "filtered": test3_depth2[0]["filtered"],
        "leaves": [
            "CC(C)(C)[C@@H](CS(C)(=O)=O)Nc1nc(Cl)ncc1F",
            "Cc1ccc(S(=O)(=O)n2cc(B3OC(C)(C)C(C)(C)O3)c3cc(F)cnc32)cc1",
        ],
    }
]

test8_n1route_leaves = [
    {
        "filtered": n1_route_idx0_out,
        "leaves": [
            "CC(C)(C)[C@H](N)CO",
            "Fc1cnc(Cl)nc1Cl",
            "CS(=O)(=O)Cl",
            "CC(O)=S",
            "CI",
            "O=S(O)OO",
            "Cc1ccc(S(=O)(=O)n2cc(B3OC(C)(C)C(C)(C)O3)c3cc(F)cnc32)cc1",
        ],
    }
]

test9_tknz_smiles = [
    ('COC(=O)c1cc2c(cc1[N+](=O)[O-])OCCO2', ['<SOS>', 'C', 'O', 'C', '(', '=', 'O', ')', 'c', '1', 'c', 'c', '2', 'c', '(', 'c', 'c', '1', '[', 'N', '+', ']', '(', '=', 'O', ')', '[', 'O', '-', ']', ')', 'O', 'C', 'C', 'O', '2', '?']),
    ('BrCCBr', ['<SOS>', 'B', 'r', 'C', 'C', 'B', 'r', '?']),
]

path_string_1 = "{'smiles':'COC(=O)c1cc2c(cc1[N+](=O)[O-])OCCO2','children':[{'smiles':'COC(=O)c1ccc2c(c1)OCCO2','children':[{'smiles':'COC(=O)c1ccc(O)c(O)c1'},{'smiles':'BrCCBr'}]},{'smiles':'O=[N+]([O-])O'}]}"
path_string_1_tknz = ['<SOS>', '{', "'smiles':", "'", 'C', 'O', 'C', '(', '=', 'O', ')', 'c', '1', 'c', 'c', '2', 'c', '(', 'c', 'c', '1', '[', 'N', '+', ']', '(', '=', 'O', ')', '[', 'O', '-', ']', ')', 'O', 'C', 'C', 'O', '2', "'", ',', "'children':", '[', '{', "'smiles':", "'", 'C', 'O', 'C', '(', '=', 'O', ')', 'c', '1', 'c', 'c', 'c', '2', 'c', '(', 'c', '1', ')', 'O', 'C', 'C', 'O', '2', "'", ',', "'children':", '[', '{', "'smiles':", "'", 'C', 'O', 'C', '(', '=', 'O', ')', 'c', '1', 'c', 'c', 'c', '(', 'O', ')', 'c', '(', 'O', ')', 'c', '1', "'", '}', ',', '{', "'smiles':", "'", 'B', 'r', 'C', 'C', 'B', 'r', "'", '}', ']', '}', ',', '{', "'smiles':", "'", 'O', '=', '[', 'N', '+', ']', '(', '[', 'O', '-', ']', ')', 'O', "'", '}', ']', '}', '?']
test10_tknz_path = [
    (path_string_1, path_string_1_tknz),
]
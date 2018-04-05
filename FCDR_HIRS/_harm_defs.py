"""Harmonisation definitions.

Definitions relating to the output of harmonisation.

In particular, the `harmonisation_parameters` dictionary contains the
relevant parameters for each satellite and channel for which harmonisation
has so far been applied in the form of:

`Dict[str, Dict[int, Dict[int, float]]]`

For example, to get a₀ for noaa18:

`harmonisation_parameters["noaa18"][12][0]`

Harmonisation parameters are derived using software developed by Ralf Quast.
"""

###############################################
###                                         ###
### AUTOMATICALLY GENERATED — DO NOT EDIT!! ###
###                                         ###
###############################################

harmonisation_parameters =  {'metopa': {1: {0: 2.9459138777326034e-15,
                1: -2.2739776101697823e-22,
                2: 0.005711424345785645},
            2: {0: -6.195721205598414e-16,
                1: -6.32609491306781e-21,
                2: 0.02336282290172156},
            3: {0: -1.189513369829004e-13,
                1: -2.715184898632757e-19,
                2: -0.03705793189072383},
            4: {0: -1.8864330570359248e-13,
                1: -3.572095149176044e-19,
                2: -0.06107017682572921},
            5: {0: -1.437127403268467e-14,
                1: 6.725605721551585e-21,
                2: 0.03393006160695815},
            6: {0: -4.738326165985468e-14,
                1: -2.3860990345722715e-20,
                2: 0.024036781299609237},
            7: {0: -2.6331525698261622e-14,
                1: 1.382165444780876e-21,
                2: 0.03487641042888461},
            8: {0: -3.675207233899877e-15,
                1: 2.6452558576820022e-21,
                2: 0.019407284903765754},
            9: {0: -2.8955689193614424e-15,
                1: -1.2086313782705723e-20,
                2: 0.016082872619687036},
            10: {0: -9.06898926360435e-15,
                 1: 2.225208754130968e-21,
                 2: 0.021542477438892507},
            11: {0: -8.396195462259682e-15,
                 1: -2.940158144492321e-20,
                 2: 0.02089964660672204},
            12: {0: -7.844590349560227e-15,
                 1: -2.633410495149053e-19,
                 2: -0.1812184351932442},
            13: {0: 4.6583576313809825e-17,
                 1: -5.620009446926258e-22,
                 2: -0.01476856151737801},
            14: {0: -7.477376175254476e-17,
                 1: -5.828731978733168e-22,
                 2: 0.007230402930668027},
            15: {0: 1.7441979200042097e-17,
                 1: -6.047386764997141e-21,
                 2: -0.1800858717283149},
            16: {0: 2.5224168357702795e-17,
                 1: 1.6088029241560773e-22,
                 2: -1.4746393566183548e-05},
            17: {0: -6.010506246413946e-18,
                 1: 3.057701821021635e-22,
                 2: 0.023384395595586378},
            18: {0: -2.415075666688314e-17,
                 1: 1.2492514884824884e-22,
                 2: 0.017868871229273338},
            19: {0: -2.3093755859618257e-17,
                 1: 6.60968650847717e-23,
                 2: 0.0161271219863739}},
 'metopb': {1: {0: -1.0326438075901344e-18,
                1: 3.581645254311237e-26,
                2: -1.9585793200225172e-06},
            2: {0: 4.662075626425774e-18,
                1: -1.442692801234667e-24,
                2: 3.347597682714817e-05},
            3: {0: 1.0502095817645573e-14,
                1: -2.7222177096234842e-21,
                2: 0.01655331505091362},
            4: {0: 1.0321350290747667e-14,
                1: -6.891170615730952e-21,
                2: 0.015299087552116616},
            5: {0: 1.1035424629313562e-16,
                1: -8.734682881740742e-23,
                2: 0.00020816057552206006},
            6: {0: 8.580420602148403e-15,
                1: -5.571688611061433e-21,
                2: 0.01710397666236089},
            7: {0: 5.814740515367723e-15,
                1: -6.919811617473798e-21,
                2: 0.013868472500666832},
            8: {0: 6.124649518494441e-16,
                1: -4.750495029643108e-22,
                2: 0.001131518629656097},
            9: {0: 1.0874503334333802e-14,
                1: -1.728646281700086e-21,
                2: 0.009453085073810864},
            10: {0: 5.3251029866543375e-15,
                 1: -4.111330105570341e-21,
                 2: 0.012029757893997904},
            11: {0: 6.733464423221369e-15,
                 1: -6.544670043253038e-22,
                 2: 0.0005023287636405707},
            12: {0: -1.2388160932295723e-15,
                 1: -3.9381137713181333e-23,
                 2: 0.00011072310993646597},
            13: {0: 2.855831935479564e-16,
                 1: 7.948315097473503e-21,
                 2: -0.001764309795164534},
            14: {0: -1.3455183048775652e-16,
                 1: 8.873766688139932e-22,
                 2: -9.26902973147112e-05},
            15: {0: 1.1752505831869147e-16,
                 1: 2.622512696703731e-21,
                 2: -0.00016343373942860132},
            16: {0: 1.891629177256318e-18,
                 1: -1.2919435350438698e-24,
                 2: 8.88764122223308e-08},
            17: {0: 1.0721080151999916e-16,
                 1: -2.7746840705140243e-22,
                 2: -8.762404727639799e-05},
            18: {0: 1.0790166741739146e-17,
                 1: -1.5955120364611098e-22,
                 2: -3.2324468212228995e-05},
            19: {0: -1.0275220341743264e-18,
                 1: -2.405819753051396e-23,
                 2: -2.2586526667535833e-05}},
 'noaa16': {1: {0: 2.1981035116281284e-17,
                1: -1.98401289305685e-24,
                2: 3.272677992403012e-05},
            2: {0: 9.4556215473339e-15,
                1: -1.600399312797981e-21,
                2: 0.014278614823386238},
            3: {0: -4.2154261971640465e-15,
                1: -2.130044906727428e-21,
                2: 0.02620809987677492},
            4: {0: 1.1136556386967663e-14,
                1: -5.7487287871042436e-21,
                2: 0.014046790275231398},
            5: {0: 4.9003079337695566e-15,
                1: -4.9602712522488035e-21,
                2: 0.008334317814677637},
            6: {0: 6.137444825478719e-15,
                1: -5.947407570740178e-21,
                2: 0.01160680979993146},
            7: {0: 4.376654480514933e-15,
                1: -4.705117263002122e-21,
                2: 0.013196167555431512},
            8: {0: 1.4524807409238381e-15,
                1: -1.807014585529545e-21,
                2: 0.0030047587707673303},
            9: {0: 1.27797068035622e-14,
                1: -6.652067203178955e-21,
                2: 0.01483888990656299},
            10: {0: 6.0153811191365745e-15,
                 1: -7.144677329972351e-21,
                 2: 0.012325982963144697},
            11: {0: 5.132506916436998e-16,
                 1: -8.979531993969339e-21,
                 2: 0.0255979248290206},
            12: {0: 9.685182346807509e-16,
                 1: 1.0455309153412308e-21,
                 2: -0.0028609740204330898},
            13: {0: 3.2025113526260156e-16,
                 1: 3.1625275495122114e-21,
                 2: 0.00036784682559598265},
            14: {0: 1.373924128761997e-16,
                 1: 1.459991922215918e-21,
                 2: -0.00016006605226943338},
            15: {0: 2.483139709388412e-16,
                 1: 1.6770150635599395e-21,
                 2: -0.0005283799258381536},
            16: {0: -1.826759791093597e-17,
                 1: 1.2413356617479875e-23,
                 2: -3.4068099848560724e-07},
            17: {0: 9.998985732448382e-17,
                 1: 8.463130925371371e-22,
                 2: 0.001160182050137345},
            18: {0: 7.216049632747504e-17,
                 1: 3.864938926656317e-22,
                 2: 3.4012810219886976e-05},
            19: {0: 3.065326234054672e-17,
                 1: 4.613269967929728e-22,
                 2: 0.0001449001064623788}},
 'noaa17': {1: {0: 2.822835748176776e-17,
                1: -7.209732123868976e-26,
                2: 4.214372697225709e-05},
            2: {0: 1.479951446492339e-14,
                1: -2.554706536352075e-22,
                2: 0.015998591925008694},
            3: {0: -2.1616400522126237e-14,
                1: -1.8336255880274066e-21,
                2: 0.03999454789177733},
            4: {0: -7.918108113192742e-15,
                1: -4.896383750313522e-21,
                2: 0.0299640392502991},
            5: {0: 7.712851406658583e-15,
                1: -3.047456577299435e-21,
                2: 0.01356229801429573},
            6: {0: -4.104000519871999e-15,
                1: 1.4113293994067366e-21,
                2: 0.01962715018930668},
            7: {0: -1.0336136903214851e-14,
                1: 1.7560374589293178e-20,
                2: 0.03541691062090254},
            8: {0: 6.538117991132209e-15,
                1: -4.491853243606799e-21,
                2: 0.01233038332494187},
            9: {0: 5.205982894813794e-15,
                1: -2.017160564286301e-21,
                2: 0.015988782207430465},
            10: {0: 3.0221351602780675e-15,
                 1: 2.3040704309692383e-21,
                 2: 0.016800679388920454},
            11: {0: -4.9718049736159916e-15,
                 1: -1.5993288471554415e-20,
                 2: 0.04531682778208925},
            12: {0: 2.244901201871615e-15,
                 1: 1.916931642552264e-21,
                 2: -0.007664174672129015},
            13: {0: 2.704535493598785e-16,
                 1: 6.585053665531251e-21,
                 2: -0.0017574100060458822},
            14: {0: 2.6617473106673236e-17,
                 1: 2.816647274118229e-22,
                 2: -5.5963592348710277e-05},
            15: {0: 2.2226156141124736e-16,
                 1: 3.1680915459597374e-21,
                 2: -0.003748542475120758},
            16: {0: -8.721917357721034e-17,
                 1: 1.1732060210952814e-23,
                 2: -5.525705908469544e-07},
            17: {0: 2.6529950843160068e-17,
                 1: -2.358751542824845e-21,
                 2: 0.002127927644189919},
            18: {0: 7.068041402228235e-17,
                 1: 1.976518643568063e-22,
                 2: 9.548812371162203e-05},
            19: {0: -3.2685950288122907e-17,
                 1: 3.5726177685099345e-22,
                 2: -0.0002458987993453486}},
 'noaa18': {1: {0: -3.903404632924674e-18,
                1: 4.681435885558904e-25,
                2: -7.158738219236094e-06},
            2: {0: 7.607186163634055e-15,
                1: -6.165219165785353e-22,
                2: 0.012029429055794941},
            3: {0: -1.7803382822049628e-14,
                1: -2.8588467659273914e-21,
                2: 0.028544798385060452},
            4: {0: 6.289830240774194e-15,
                1: -5.926147685869203e-21,
                2: 0.01955320956466878},
            5: {0: 1.3200055423380618e-15,
                1: -1.0121946555723475e-21,
                2: 0.0020916239544851896},
            6: {0: 5.939745317045895e-15,
                1: -4.3557984605228576e-21,
                2: 0.010302517127315563},
            7: {0: 7.034199958869325e-15,
                1: -7.110110876761916e-21,
                2: 0.014187319557166879},
            8: {0: 1.1184999311145486e-15,
                1: -8.061403609807594e-22,
                2: 0.0015193018340127881},
            9: {0: 7.832210595605091e-15,
                1: -1.2139068421909917e-21,
                2: 0.006921973168562212},
            10: {0: 4.26107417855422e-15,
                 1: -3.5992159894846386e-21,
                 2: 0.0076855781858063054},
            11: {0: 6.467895963248214e-15,
                 1: -3.3030225860034415e-21,
                 2: 0.01086517895555592},
            12: {0: 2.202037365265115e-15,
                 1: -9.419043269709739e-23,
                 2: 0.00020169582781055583},
            13: {0: 4.46572586566355e-16,
                 1: 2.649410547490992e-21,
                 2: -0.00015400600528094307},
            14: {0: -6.986055410507437e-18,
                 1: 8.143639051906336e-22,
                 2: -3.666226666637885e-05},
            15: {0: 3.052344535130292e-16,
                 1: 1.781341135553704e-21,
                 2: -5.142128967065682e-05},
            16: {0: 7.100605957992629e-17,
                 1: -4.170804149441224e-25,
                 2: 1.1464325794880974e-08},
            17: {0: 2.984826402725178e-17,
                 1: -1.432006421257944e-21,
                 2: 0.00015196251993818122},
            18: {0: 3.6399921710270946e-17,
                 1: 9.386296208605088e-23,
                 2: -1.3402679789059531e-05},
            19: {0: 6.481254643278746e-18,
                 1: -5.521898217024104e-23,
                 2: 4.143125812855333e-06}},
 'noaa19': {1: {0: -1.9502316362445616e-18,
                1: 1.0965972020782942e-25,
                2: -3.3316503653198854e-06},
            2: {0: 1.6493651237149987e-15,
                1: -1.7197024532125369e-22,
                2: 0.0026443563947484666},
            3: {0: 3.689343298553664e-16,
                1: -3.763950131653173e-21,
                2: 0.01310301540108312},
            4: {0: 6.635992844959846e-15,
                1: -7.014993179252888e-21,
                2: 0.016063750385092047},
            5: {0: 1.1164899399278539e-15,
                1: -8.019550326485014e-22,
                2: 0.0018914724079368383},
            6: {0: 8.583888976632662e-15,
                1: -9.389400599178043e-21,
                2: 0.0157402498660863},
            7: {0: 6.843977813190224e-15,
                1: -8.260564956961734e-21,
                2: 0.013425528256907434},
            8: {0: 1.7052968763433566e-16,
                1: -1.8154719613007173e-22,
                2: 0.0002456532636415776},
            9: {0: 7.143167855162613e-15,
                1: -1.5214824737650761e-21,
                2: 0.005555044950893242},
            10: {0: 2.212307012723523e-15,
                 1: -2.684227221464015e-21,
                 2: 0.0039833417076406965},
            11: {0: 3.654960873348135e-15,
                 1: -4.47437788935762e-21,
                 2: 0.006305140706534277},
            12: {0: 2.9748420781270013e-15,
                 1: -4.370962086584327e-22,
                 2: 0.0006987442409026323},
            13: {0: 3.838459480417321e-16,
                 1: 5.577120653977014e-21,
                 2: -0.00032872839300781013},
            14: {0: -3.233614534369687e-18,
                 1: -2.239942826974507e-22,
                 2: 9.789783239687e-06},
            15: {0: 3.9961277129899807e-16,
                 1: 4.582851988196585e-21,
                 2: -0.00020612036214553512},
            16: {0: 6.7134169791399456e-18,
                 1: -4.283419128176311e-25,
                 2: 3.4624503346966524e-08},
            17: {0: 6.009239498448829e-17,
                 1: -9.68295445965095e-22,
                 2: 9.62496538922928e-05},
            18: {0: 4.7151389891368395e-17,
                 1: -2.4575606099457227e-24,
                 2: 1.976131709917894e-07},
            19: {0: -1.5000482169255413e-17,
                 1: 5.755896026503849e-23,
                 2: -3.2452367703937603e-06}}}

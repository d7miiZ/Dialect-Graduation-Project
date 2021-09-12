================================================================================
DART: A Large Dataset of Dialectal Arabic Tweets (c) 2018 Qatar University
================================================================================
Release date: May 8th, 2018
For any questions, please contact Tamer Elsayed (telsayed@qu.edu.qa) or any of the authors of DART paper:
Israa Alsarsour, Esraa Mohamed, Reem Suwaileh, Tamer Elsayed
Computer Science and Engineering Department, Qatar University, Doha, Qatar
{ia1205702, em1205267, reem.suwaileh, telsayed}@qu.edu.qa

NOTE: If you use our dataset, kindly cite the paper in your work. Thanks!
@inproceedings{alsarsour2018dart,
  title={DART: A Large Dataset of Dialectal Arabic Tweets},
  author={Alsarsour, Israa and Mohamed, Esraa and Suwaileh, Reem and Elsayed, Tamer},
  booktitle={Proceedings of the 11th International Conference on Language Resources and Evaluation (LREC'18},
  pages={3666--3670},
  booktitle={LREC},
  year={2018}
}

================================================================================
Format:
================================================================================
This release contains the following data:
	1. 24,280 Arabic labeled tweets and 1,657 gold questions (under "cf-data" directory)
	2. 1,108 Arabic dialectal phrases (222 per dialect on avg) used for tracking over Twitter stream (under "tracking-pharses" directory)
	3. 500 tweets that are labeled by native speakers (100 per dialect) to evaluate CF's labels accuracy (under "eval-acc" directory)

The size of all data is about 3.95MB. You can find detailed statistics by reading the paper cited above (DART: A Large Dataset of Dialectal Arabic Tweets)

Each dialect group has its own file that contains data belongs to it. Files are formatted as follows:
	1. cf-data: 
		a.	Labeled data: <score(/3)>\t<tweet_ID>\t<tweet_text>
			-	score: the majority score for each tweet of 3 annotators
			-	tweet_text: the original textual content of the labelled tweet 
		b.	Gold questions: <label>\t<tweet_text>
			-	label: label of native speacker
			-	tweet_text: the original textual content of the labelled tweet 
	2. eval-acc: <label>\t<tweet_text>
		-	label: label of native speacker
		-	tweet_text: the original textual content of the labelled tweet 
	3. tracking-phrases: <tracking-phrase>

================================================================================
Dialect Groups:
================================================================================
Egyptian (EGY): Egypt
Gulf (GLF): UAE, KSA, Bahrain, Kuwait, and Qatar
Levantine (LEV): Jordan, Palestine, Syria, and Lebanon
Iraqi (IRQ): Iraq
Maghrebi (MGH): Morocco, Algeria, and Tunisia
================================================================================
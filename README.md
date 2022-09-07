2425789f Antonius Clarence Fionaldi

Software is found on
- Source Code Mongodb
- Executable & Source Code txt

Source Code Mongodb contains 3 sources files

- streamCrawler
Connects to a mongo database then harvests and stores processed Twitter objects using the Streaming API/REST API.
- downloadMedia
Connects to a mongo database and downloads the first image and video file.
- dataAnalyzer
Connects to a mongo database and analyzes and clusterize the processed twitter data.

Executable & Source Code txt contains 4 source files, 2 executables, and one twitter data text file.

- twitterData
contains 5 minutes worth of twitter objects stored on a .txt file
- run
runs dataAnalyzertxt, downloadMediatxt, and clusteringtxt
- dataAnalyzertxt.py
Harvests data from twitterData and analyzes them (count the occurance of retweets etc)
- dataAnalyzertxt.exe
The executable version fo dataAnalyzertxt.py
- downloadMediatxt.py
Harvests data from twitterData and downloads the first image and video file.
- downloadMediatxt.exe
The executable version fo downloadMediatxt.py
- clusteringtxt
Harvest data from twitterData and clusters them.


 

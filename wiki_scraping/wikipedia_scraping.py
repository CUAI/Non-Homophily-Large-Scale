from collections import deque
from collections import defaultdict
# import torch
import time
import requests
import pickle
from os import path

# This file scrapes wikipedia using the Wikipedia API, starting from the Hilbert Space page.
# At regular intervals, the new edges are dumped using pickle, so that the script can be restarted
# if Wikipedia blocks the requests after some time.

# The script also has mechanisms to pause for increasing intervals of time and resumes, if Wikipedia blocks
# requests

def send_req(S, URL, BASE_PARAMS):
	try:
		R = S.get(url=URL, params=BASE_PARAMS)
		return (R, True)
	except:
		print("\n\n")
		print("ERROR OCCURRED: SLEEPING FOR AN HOUR")
		print()
		time.sleep(3600)
		R, ans = send_req(S, URL, BASE_PARAMS)
		while (not ans):
			print("trying again")
			R, ans = send_req(S, URL, BASE_PARAMS)
		return (R, True)

def graph():
		URL = "https://en.wikipedia.org/w/api.php"
		BASE_PARAMS = {
				"action": "query",
				"format": "json",
				"prop": "links|info|pageviews|extracts",
				"indexpageids": 1,
				"titles": "",
				"redirects": 1,
				"pllimit": "500",
				"inprop": "url",
				"exintro": 1,
				"explaintext": 1
		}
		S = requests.Session()
		start_page = "Hilbert space"
		

		text = open("pagetext.obj", "ab+")
		edges = open("edges.obj", "ab+")
		if (path.exists("checkpoint.obj")):
			print("unpickling...")
			c = open("checkpoint.obj", "rb")
			visited = pickle.load(c)
			views = pickle.load(c)
			indices = pickle.load(c)
			q = pickle.load(c)
			iters = pickle.load(c)
		else:
			q = deque()
			q.append(start_page)
			iters = 1999999
			visited = set()
			views = defaultdict(int)
			indices = {}
			indices[start_page] = 0
			visited.add(start_page)
		start = iters
		prefix = "https://en.wikipedia.org/wiki/"
		while (q and iters > 0):
				if (iters % 200 == 0):
						print("iters done: ", start - iters)
						print("iters left:", iters)
						time.sleep(2)
				if (iters % 500 == 0):
						time.sleep(10)
				if (iters % 1000 == 0):
						print("iters done: ", start - iters)
						with open("checkpoint.obj", "wb") as c:
								pickle.dump(visited, c)
								pickle.dump(views, c)
								pickle.dump(indices, c)
								pickle.dump(q, c)
								pickle.dump(iters, c)
						time.sleep(20)
				if (iters % 10000 == 0):
						print("iters done: ", start - iters)
						print("sleeping")
						time.sleep(600)
				if (iters % 18000 == 0):
						print("iters done: ", start - iters)
						print("sleeping for 20 mins")
						time.sleep(1200)
				page = q.popleft()

				BASE_PARAMS["titles"] = page
				
				R, _ = send_req(S, URL, BASE_PARAMS)
				DATA = R.json()

				while ("errors" in DATA):
						print("\n\n")
						print("ERROR OCCURRED: SLEEPING FOR AN HOUR")
						print()
						time.sleep(3600)
						R, _ = send_req(S, URL, BASE_PARAMS)
						DATA = R.json()

				page_id = DATA["query"]["pageids"][0]

				if "missing" not in DATA["query"]["pages"][page_id] and "links" in DATA["query"]["pages"][page_id]:
						LINKS = DATA["query"]["pages"][page_id]["links"]
						views[page] = sum(
								filter(None, DATA["query"]["pages"][page_id]["pageviews"].values()))

						pickle.dump((indices[page], DATA["query"]
												 ["pages"][page_id]["extract"]), text)
						iters -= 1

						while ("continue" in DATA):
								print(page)
								print("making additional request for same page")
								BASE_PARAMS["plcontinue"] = DATA["continue"]["plcontinue"]
								BASE_PARAMS["continue"] = DATA["continue"]["continue"]
								R, _ = send_req(S, URL, BASE_PARAMS)
								DATA = R.json()
								try:
										LINKS += DATA["query"]["pages"][page_id]["links"]
								except:
										del BASE_PARAMS["plcontinue"]
										del BASE_PARAMS["continue"]
										break
								del BASE_PARAMS["plcontinue"]
								del BASE_PARAMS["continue"]

						for l in LINKS:
								if l["ns"] == 0:
										t = l["title"]
										pickle.dump((page, t), edges)
										# graph[page].add(t)

										if t not in visited:
												q.append(t)
												# generating numerical key for each new article encountered
												indices[t] = len(visited)
												visited.add(t)
				else:
						print("MISSING")
						print(page)
		text.close()
		edges.close()
		print(views)


graph()

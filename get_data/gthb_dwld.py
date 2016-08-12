import github as gthb

inst = gthb.Github()


languages  = ['Python', 'PHP', 'Ruby', 'Javascript', 'Haskell']
extensions = ['.py', '.php', '.rb', '.js', '.hs']

with open('urls.txt','w') as f:

	for ind, lang in enumerate(languages):	
		repos = inst.search_repositories(lang, language=lang, sort='stars', order='desc')

		size = 0
		i = 0
		while size < 3e5:

			url = repos[i].html_url	
			size += repos[i].size
			i+=1

			f.write(lang+','+url+','+extensions[ind]+'\n')	




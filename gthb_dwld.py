import github as gthb

inst = gthb.Github()


languages = ['Python', 'PHP', 'Ruby', 'Javascript', 'Haskell']


lang_urls = []
for lang in languages:
	repos = inst.search_repositories(lang, language=lang, sort='stars', order='desc')

	size = 0
	i = 0
	urls = []
	while size < 1e3:

		urls.append(repos[i].html_url)	
		size += repos[i].size
		i+=1

	lang_urls.append(urls)


#! /usr/bin/env bash

for line in $(cat urls.txt) 
do

	IFS=',' read -a vars <<< "${line}"
	dir_name=${vars[0]}
	url=${vars[1]}/tarball/master
	extension=${vars[2]}
	
        mkdir $dir_name
	cd $dir_name
	
	curl -L $url > download.tar.gz

	mkdir download

	tar -xzvf download.tar.gz -C download/
	
	find download -name *$extension -exec mv {} . \;

	rm -rf download

	rm download.tar.gz

	cat *$extension >> ${extension}.txt

	rm *$extension

	cd ..

done

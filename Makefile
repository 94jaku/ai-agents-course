node_modules:
	npm install

task-%: node_modules
	$(MAKE) -C $@ run

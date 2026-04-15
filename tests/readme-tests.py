Explanation of tests. 

There are different types of tests: 
- Unit tests: 
	Testing a single function, making sure it works.
- Integration tests: 
	Testing multiple components working together.  
	Examples: loading an image and finding the centerpoint.
- System tests: 
	Testing the entire system, end-to-end. Often requires hardware
	Example: Start software, take an image, determine position
- Regression tests: 
	Ensures performance doesn't degrade over time: 
	Examples: Previously fixed bugs, new functions slow down performance.
- Hardware tests: 
	Tests hardware specifically.  Often separate from others because equipment is required. 
	Example: 
- Acceptance tests: 
	Validates that the system meets requirements before being shipped
	May result in a certificate
	Example: Quality certificate before shipment.
- Perforamnce Tests: 
	Ensures system meets specific benchmarks. 
	Example: Center detection runs <20 us

In addition the following folder is necessary: 

- Assets: 
	These are the fixed exaples that let the tests run smoothy. 
	Subfolders: Images (source images), 
		Expected (expected outputs)

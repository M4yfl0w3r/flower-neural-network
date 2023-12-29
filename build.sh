#!/bin/lua

os.execute("mkdir -p build")
os.execute("premake5 --file=premake5.lua gmake")
os.execute("make")
os.execute("./build/Debug/Net")
os.execute("rm Makefile")
#!/bin/lua

os.execute("mkdir -p assets/iris")

function file_exists(name)
    local f = io.open(name, 'r')
    if f ~= nil then 
        io.close(f)
        return true 
    else 
        return false 
    end
end

local dataset_path = 'iris_dataset'

if not file_exists('assets/iris/iris.data') then
    local dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    os.execute('wget ' .. dataset_url .. ' -O ' .. 'iris.data')

    os.execute('mv iris.data assets/iris/')
end

os.execute("mkdir -p build")
os.execute("premake5 --file=premake5.lua gmake")
os.execute("make")
os.execute("./build/Debug/Net")
os.execute("rm Makefile")


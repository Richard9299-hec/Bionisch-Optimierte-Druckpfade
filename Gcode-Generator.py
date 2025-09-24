import copy
from shapely.geometry import Point, Polygon
import geopandas as gpd
import matplotlib.pyplot as plt
input="old_gcode.gcode"
output="new_gcode.gcode"
stress_lines="stress_lines.txt"
extrusion = 0.1
mirror_fuse_distance= 2.75

def get_layerheight(header):
    index=0
    while index< len(header):
        line = header[index]
        if line[:14]== ";Layer height:":
            return  float(line[14:])
        index +=1
    return 0

def read_in_gcode():
    lines=[]
    data=open(input)
    line=data.readline()
    while line != "":
        lines.append(line)
        line=data.readline()
    return lines

def get_E(line):
    return float(line.split("E")[1])

def replace_E(line,newE:float):
    index = line.find('E')
    new_line= line[:index]+"E"+str(newE)+"\n"
    return new_line

def read_in_lines():
    lines = []
    data = open(stress_lines,"r")
    line = data.readline()
    help = []
    while line != "":
        if "#" in line:
            if len(help)>0:
                lines.append(help)
                help=[]
            line = data.readline()
            continue
        if line == "\n":
            line = data.readline()
            continue
        else:
            line=line.split(" ")
            help.append([float(line[0]),float(line[1])])
            line = data.readline()
    lines.append(help)
    data.close()
    return lines

def jump_to_gcode(x:float,y:float):
    return'G0 X'+str(x)+' Y'+str(y)+'\n'

def generate_infill_gcode(xa,ya,xb,yb,height):
    e = extrusion*height*((((float(xa)-float(xb))**2)+((float(ya)-float(yb))**2))**0.5)
    return ('G1 X'+str(xb)+' Y'+str(yb)+' E'+str(e)[:7]+'\n')

def get_center(gcode): # gets the center of original object
    x = [0,0]
    y = [0,0]
    index =0
    line = gcode[index]
    while line[:6] != ";TYPE:":
        index +=1
        line = gcode[index]
    while line[:7] != ";LAYER:":
        if "G" in line and "X" in line:
            cords = get_coords(line)
            if cords[0] < x[0]:
                x[0] = cords[0]
            if cords[0] > x[1]:
                x[1] = cords[0]
            if cords[1] < y[0]:
                y[0] = cords[0]
            if cords[1] > y[1]:
                y[1] = cords[1]
            if x[0] ==0:
                x[0] = cords[0]
            if y[0] == 0:
                y[0] = cords[1]
        index +=1
        line = gcode[index]
    return [(x[0]+x[1])/2,(y[0]+y[1])/2]

def get_coords(line): #for gcode
    line = line.split("X")[1]
    line = line.split("Y")
    x=float(line[0])
    line = line[1]
    if " " in line:
        y=float(line.split(" ")[0])
    else:
        y = float(line)
    return[x,y] # flaots

def get_center_lines(lines):
    y = [0, 0]
    index_1 = 0
    index_2 = 0
    while(index_1<len(lines)):
        while(index_2<len(lines[index_1])):
            if y[0] == 0:
                y[0] = lines[index_1][index_2][1]
            if y[1] == 0:
                y[1] = lines[index_1][index_2][1]
            if y[0] > lines[index_1][index_2][1]:
                y[0] = lines[index_1][index_2][1]
            if y[1] < lines[index_1][index_2][1]:
                y[1] = lines[index_1][index_2][1]
            index_2 += 1
        index_1 += 1
    return (float(y[0]) + float(y[1])) / 2

def write_infill(lines,layerheight):
    index_1=0
    infill=[]
    while index_1 < len(lines):
        infill.append("G0 F2700 E-1\n")
        infill.append(jump_to_gcode(lines[index_1][0][0],(lines[index_1][0][1])))
        infill.append("G0 F2700 E 1\n")
        index_2 = 1
        while index_2 < len(lines[index_1]):
            xa= lines[index_1][index_2-1][0]
            ya= lines[index_1][index_2-1][1]
            xb= lines[index_1][index_2][0]
            yb= lines[index_1][index_2][1]
            infill.append(generate_infill_gcode(xa,ya,xb,yb,layerheight))
            index_2 +=1
        index_1 +=1
    return infill

def write_to_relative_extrusion(gcode):
    index=0
    help = gcode
    e1=0
    while index < len(gcode):
        line = help[index]
        if "G" in line and "E" in line:
            if e1==0:
                e1 = get_E(line)
            else:
                e2= get_E(line)
                line = replace_E(line,e2-e1)
                help[index]=line
                e1=e2
        index +=1
    return help

def replace_infill(gcode,infill):
    start_index=0
    end_index=0
    index =0
    help = gcode
    infill_count=0
    removed =0
    while index<len(help):
        line = help[index]
        if line[:9] == ";TYPE:FIL":
            infill_count +=1
        index +=1
    index =0
    while  removed < infill_count:
        while start_index == 0 or end_index == 0 and index < len(help):
            line = help[index]
            #print(line)
            if line[:9] == ";TYPE:FIL":
                start_index = index + 1
            if line[:5] == ";MESH"and start_index!=0:
                end_index = index
            index +=1
        del help[start_index:end_index]
        removed +=1
        index = start_index + 3
        start_index=0
        end_index=0
    replaced =0
    index = 0
    while replaced < infill_count:
        line = help[index]
        if line[:9] == ";TYPE:FIL":
            help[index+1:index+1]=infill
            replaced += 1
        index += 1
    return help

def cut_gcode(gcode): #important
    start = 0
    end = len(gcode)-1
    line = gcode[start]
    while line[:13] != ";LAYER_COUNT:":
        start +=1
        line = gcode[start]
    line = gcode[end]
    while line[:14] != ";TIME_ELAPSED:":
        end -= 1
        line = gcode[end]
    return[gcode[0:start],gcode[start:end],gcode[end:len(gcode)]]

def write_to_absolute_extrusion(gcode):
    index = 0
    help = gcode
    e1 = 0
    while index < len(gcode):
        line = help[index]
        if "G" in line and "E" in line:
            if e1 == 0:
                e1 = get_E(line)
            else:
                e2 = get_E(line)
                line = replace_E(line, e2+e1)
                help[index] = line
                e1 += e2
        index += 1
    return help

def fuse_gcode(header,main_gcode,end):
    result = open(output,'w')
    gcode = header
    gcode.extend(main_gcode)
    gcode.extend(end)
    index =0
    while index < len(gcode):
        result.write(gcode[index])
        index +=1

def mirror_line(line):
    index = 0
    while index < len(line):
        line[index][0] *= -1
        index +=1

def mirror_lines(lines):
    temp = copy.deepcopy(lines)
    for elem in temp:
        if elem[0][0]<=mirror_fuse_distance:
            tmp = copy.deepcopy(elem)
            mirror_line(tmp)
            elem.reverse()
            elem.extend(tmp[1:])
        elif elem[len(elem)-1][0]<=mirror_fuse_distance:
            tmp = copy.deepcopy(elem)
            mirror_line(tmp)
            tmp.reverse()
            elem.extend(tmp[1:])
    index =len(lines)-1
    while index >= 0:
        if len(temp[index])==len(lines[index]):
            line = copy.deepcopy(lines[index])#
            mirror_line(line)
            temp.append(line)
        index -=1
    return temp

def move_lines(lines,dx,dy):
    index_1 =0
    while index_1 < len(lines):
        index_2 =0
        while index_2 < len(lines[index_1]):
            lines[index_1][index_2][0] += dx
            lines[index_1][index_2][1] += dy
            index_2 += 1
        index_1 += 1

def filament_used(gcode):
    index =1
    while index <= len(gcode):
        line = gcode[len(gcode) - index]
        if "E" in line and "G" in line:
            e = get_E(line)
            return str(round(e*3.14*0.875**2*1.24*0.001,1))+" gramms of Pla used."
        index += 1
    return 0

def cut_lines(lines,boundaries,show):
    boundaries[1].append(copy.deepcopy(boundaries[1][0]))
    polygon = Polygon(boundaries[1],[boundaries[0],boundaries[2]]) #
    gdf = gpd.GeoDataFrame([1], geometry=[polygon])
    gdf.plot(facecolor='lightblue', edgecolor='black')
    plt.title("BremsbÃ¼gel")
    plt.xlabel("X-Koordinate")
    plt.ylabel("Y-Koordinate")
    polygon = polygon.buffer(0) #nur zur sicherheit
    new_lines = []
    new_lines.append([])
    for line in lines:
        for point in line:
            if polygon.covers(Point(point)):
                new_lines[len(new_lines)-1].append(copy.deepcopy(point))
            else:
                new_lines.append([])
        new_lines.append([])
    new_lines = [elem for elem in new_lines if len(elem)>0]
    if show:
        plt.show()
    return new_lines

def get_bounding_box(gcode):
    boundaries = [[]]
    index =0
    boundary_index =0
    line = gcode[index]
    while line[:16] != ";TYPE:WALL-INNER":
        index += 1
        line = gcode[index]
    index += 1
    line = gcode[index]
    while line[0] != ";":
        if "E" in line:
            if get_E(line) < 0:
                boundary_index +=1
                boundaries.append([])
        if "E" in line and "X" in line:
            boundaries[boundary_index].append(copy.deepcopy(get_coords(line)))
        index +=1
        line = gcode[index]
    return boundaries

def main():
    gcode = read_in_gcode()
    lines = read_in_lines()
    lines = mirror_lines(lines)
    move_lines(lines,get_center(gcode)[0],112.5)
    data = cut_gcode(gcode)
    gcode = data[1]
    layer_height = get_layerheight(data[0])
    gcode = write_to_relative_extrusion(gcode)
    lines = cut_lines(lines,get_bounding_box(gcode),False)
    infill = write_infill(lines, layer_height)
    gcode = replace_infill(gcode,infill)
    gcode = write_to_absolute_extrusion(gcode)
    fuse_gcode(data[0], gcode, data[2])
    print(filament_used(gcode))

main()


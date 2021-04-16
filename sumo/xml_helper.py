import xml.etree.ElementTree as ET


ONEWAY_DICT = {'yes': True, 
                '-1': True, 
                '1': True, 
                'reversible': True,
                'no': False,
                '0': False
    }
# TODO: reversible, alternating: https://wiki.openstreetmap.org/wiki/Tag:oneway%3Dreversible


def indent(elem, level=0):
    i = "\n" + level * "\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
            
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

    return

def update_element_attrib(item, key, val, log=False):
    def _print():
        print({tag.get('k'): tag.get('v') for tag in item.findall('tag')})
    
    tags = [tag.get('k') for tag in item.findall('tag')]
    if log: _print()
    
    # 判断是否为双向道路，若是则需要将val/2  # <tag k="oneway" v="yes" />
    if key =='lanes':
        oneway_att = False
        if 'oneway' in tags:
            element = item.findall('tag')[tags.index('oneway')]
            oneway_att = ONEWAY_DICT[element.get('v')]
        
        if not oneway_att and isinstance(val, int):
            val = val *2

    if key in tags:
        element = item.findall('tag')[tags.index(key)]
        element.set('v', str(val))
        if log: _print()
        return True

    lanes_info = ET.Element('tag', {'k': key, 'v': str(val)})
    item.append(lanes_info)
    if log: _print()

    return True


def print_elem(elem, indent='', print_child=True):
    print(indent, {i: elem.get(i) for i in elem.keys()}) 
    if not print_child:
        return
    
    for i in elem.getchildren():
        print_elem(i, indent+'\t', print_child)    
    return 

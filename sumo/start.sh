export SUMO_HOME=/usr/share/sumo
# netconvert
/usr/share/sumo/bin/netconvert  -t /usr/share/sumo/data/typemap/osmNetconvert.typ.xml --geometry.remove --roundabouts.guess --ramps.guess -v --junctions.join --tls.guess-signals --tls.discard-simple --tls.join --output.original-names --junctions.corner-detail 5 --output.street-names --tls.default-type actuated --osm-files osm_bbox.osm.xml --keep-edges.by-vclass passenger -o osm.net.xml
# polyconvert
/usr/share/sumo/bin/polyconvert -v --osm.keep-full-type --type-file /usr/share/sumo/data/typemap/osmPolyconvert.typ.xml --osm-files osm_bbox.osm.xml -n osm.net.xml -o ./osm.poly.xml --save-configuration osm.polycfg
/usr/share/sumo/bin/polyconvert -c osm.polycfg
# randomTrips
python /usr/share/sumo/tools/randomTrips.py -n osm.net.xml --fringe-factor 5  -o osm.passenger.trips.xml -e 3600 --vehicle-class passenger --vclass passenger --prefix veh --min-distance 300 --trip-attributes "departLane=\"best\"" --fringe-start-attributes "departSpeed=\"max\"" --allow-fringe.min-length 1000 --lanes --validate
/usr/share/sumo/bin/duarouter -n ./osm.net.xml -r ./osm.passenger.trips.xml --ignore-errors --begin 0 --end 3600 --no-step-log --no-warnings -o routes.rou.xml
/usr/share/sumo/bin/duarouter -n ./osm.net.xml -r ./osm.passenger.trips.xml --ignore-errors --begin 0 --end 3600 --no-step-log --no-warnings -o ./osm.passenger.trips.xml.tmp --write-trips
# view
echo """<viewsettings>\n\t<scheme name=\"standard\"/>\n\t<delay value=\"30\"/>\n</viewsettings>""" > osm.view.xml
# create config
/usr/share/sumo/bin/sumo -n osm.net.xml --gui-settings-file osm.view.xml --duration-log.statistics --device.rerouting.adaptation-interval 10 --device.rerouting.adaptation-steps 18 -v --no-step-log --save-configuration osm.sumocfg --ignore-route-errors -r osm.passenger.trips.xml -a osm.poly.xml

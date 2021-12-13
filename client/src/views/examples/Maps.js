import React, { useRef, useEffect, useState, useCallback } from 'react';


// reactstrap components
import { Card, Container, Row } from "reactstrap";

// core components
import Header from "components/Headers/Header.js";

import mapboxgl from '!mapbox-gl'; // eslint-disable-line import/no-webpack-loader-syntax

import "./legend.css";

 
mapboxgl.accessToken = 'pk.eyJ1IjoidmlnbmVzaDExIiwiYSI6ImNrZ2FqZmJiMzAzaTYyeW1jaXZkbnF0ZnEifQ.yP7CY04e82ARmeSZBrw_ug';

const layers = [
  'Dining top rated',
  'Delivery top rated',
];
const colors = [
  '#6cd6fb',
  'orange',
];


const MapWrapper = () => {

  const mapContainer = useRef(null);
  const map = useRef(null);

  useEffect(() => {
    if (map.current) return; // initialize map only once
    map.current = new mapboxgl.Map({
    container: mapContainer.current,
    style: 'mapbox://styles/mapbox/streets-v11',
    center: [77.5946, 12.9716],
    zoom: 11
    });

    return () => map.current.remove();

  }, []);

  const fetchData = useCallback((type) => {
    if(type === 'dining'){
      return fetch("http://127.0.0.1:4200/map_dining", {
          method: 'get',
          headers: {
            'Accept': 'application/json, text/plain, */*',
            'Content-Type': 'application/json'
          },
        })
        .then(data => data.json())
    }
    else{
      return fetch("http://127.0.0.1:4200/map_delivery", {
          method: 'get',
          headers: {
            'Accept': 'application/json, text/plain, */*',
            'Content-Type': 'application/json'
          },
        })
        .then(data => data.json())
    }
  }, []);

  useEffect(() => {
    if (!map.current) return; // Waits for the map to initialise
   
   const results1 = fetchData('dining');
   results1.then((res) => {
      let rank = 1;
      res.map((loc) => {
        //console.log(loc);
        let myLatlng = new mapboxgl.LngLat(loc.lng, loc.lat);

        // make a marker for each feature and add it to the map
        new mapboxgl.Marker()
          .setLngLat(myLatlng)
          .setPopup(
            new mapboxgl.Popup({ offset: 25 }) // add popups
              .setHTML('<p>' + loc.name + '</p>' + '<p>' + "dining rank: " + rank + '</p>' + '<p>' + loc.dining_rating + '</p>' + '<p>'  + "address: " + loc.locality_address + '</p>')
          )
          .addTo(map.current);
          rank ++;
   });
      })
  
  const results2 = fetchData('delivery');
  results2.then((res) => {
      let rank = 1;
      res.map((loc) => {
        //console.log(loc);
        let myLatlng = new mapboxgl.LngLat(loc.lng, loc.lat);

        // make a marker for each feature and add it to the map
        new mapboxgl.Marker({ "color": "orange"})
        .setLngLat(myLatlng)
        .setPopup(
          new mapboxgl.Popup({ offset: 25 }) // add popups
            .setHTML('<p>' + loc.name + '</p>' + '<p>' + "delivery rank: " + rank + '</p>' + '<p>' + loc.delivery_rating + '</p>' + '<p>' + "address: " + loc.locality_address + '</p>')
        )
        .addTo(map.current);
        rank ++;
  });
      })
  
      const legend = document.getElementById('legend');

      layers.forEach((layer, i) => {
        const color = colors[i];
        const item = document.createElement('div');
        const key = document.createElement('span');
        key.className = 'legend-key';
        key.style.backgroundColor = color;
      
        const value = document.createElement('span');
        value.innerHTML = `${layer}`;
        item.appendChild(key);
        item.appendChild(value);
        legend.appendChild(item);
      });
  

 }, [fetchData]);

  return (
    <>
      <div
        style={{ height: `600px` }}
        className="map-canvas"
        id="map-canvas"
        ref={mapContainer}
      ></div>
      <div class='map-overlay' id='legend'></div>
    </>
  );
};

const Maps = () => {
  return (
    <>
      <Header />
      {/* Page content */}
      <Container className="mt--7" fluid>
        <Row>
          <div className="col">
            <Card className="shadow border-0">
              <MapWrapper />
            </Card>
          </div>
        </Row>
      </Container>
    </>
  );
};

export default Maps;

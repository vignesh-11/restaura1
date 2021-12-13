import React, { useRef, useEffect, useState, useCallback } from 'react';
// node.js library that concatenates classes (strings)
import classnames from "classnames";
// javascipt plugin for creating charts
import Chart from "chart.js";
// react plugin used to create charts
import { Line, Bar } from "react-chartjs-2";
// reactstrap components
import {
  Button,
  Card,
  CardHeader,
  CardBody,
  NavItem,
  NavLink,
  Nav,
  Progress,
  Table,
  Container,
  Row,
  Col,
} from "reactstrap";

// core components
import {
  chartOptions,
  parseOptions,
  //chartExample1,
} from "variables/charts.js";

import Header1 from "components/Headers/Header1.js";
import mapboxgl from '!mapbox-gl'; // eslint-disable-line import/no-webpack-loader-syntax
mapboxgl.accessToken = 'pk.eyJ1IjoidmlnbmVzaDExIiwiYSI6ImNrZ2FqZmJiMzAzaTYyeW1jaXZkbnF0ZnEifQ.yP7CY04e82ARmeSZBrw_ug';

var colors = {
  gray: {
    100: "#f6f9fc",
    200: "#e9ecef",
    300: "#dee2e6",
    400: "#ced4da",
    500: "#adb5bd",
    600: "#8898aa",
    700: "#525f7f",
    800: "#32325d",
    900: "#212529",
  },
  theme: {
    default: "#172b4d",
    primary: "#5e72e4",
    secondary: "#f4f5f7",
    info: "#11cdef",
    success: "#2dce89",
    danger: "#f5365c",
    warning: "#fb6340",
  },
  black: "#12263F",
  white: "#FFFFFF",
  transparent: "transparent",
};

var initOptions = Chart;


const MapWrapper = ({map}) => {

  const mapContainer = useRef(null);
  //const map = useRef(null);

  useEffect(() => {
    if (map.current) return; // initialize map only once
    map.current = new mapboxgl.Map({
    container: mapContainer.current,
    style: 'mapbox://styles/mapbox/streets-v11',
    center: [77.5946, 12.9716],
    zoom: 10
    });

    return () => map.current.remove();

  }, []);

  return (
    <>
      <div
        style={{ height: `600px` }}
        className="map-canvas"
        id="map-canvas"
        ref={mapContainer}
      ></div>
    </>
  );
};
const Index = (props) => {
  const [rest, setRest] = useState({
    name:"",
    dining_rating:0,
    dining_review_count:0,
    delivery_rating:0,
    delivery_review_count:0,
    locality_name:"",
    locality_address:"",
    cuisine:"",
    cost:"",
    "dining or delivery":"",
    performance_dining: [],
    performance_delivery: [],});
  const [activeNav, setActiveNav] = useState(1);
  const [chartExample1Data, setChartExample1Data] = useState("data1");
  const map = useRef(null);
  const chartExample1 = useRef({
    options: {
      scales: {
        yAxes: [
          {
            gridLines: {
              color: colors.gray[900],
              zeroLineColor: colors.gray[900],
            },
            ticks: {
              callback: function (value) {
                if (!(value % 1)) {
                  return value + ".00";
                }
              },
            },
          },
        ],
      },
      tooltips: {
        callbacks: {
          label: function (item, data) {
            var label = data.datasets[item.datasetIndex].label || "";
            var yLabel = item.yLabel;
            var content = "";
  
            if (data.datasets.length > 1) {
              content += label;
            }
  
            content +=  yLabel;
            return content;
          },
        },
      },
    },
    data1: {
        labels: [],
        datasets: [
          {
            label: "Performance",
            data: [],
          },
        ],
    },
    data2: {
        labels: [],
        datasets: [
          {
            label: "Performance",
            data: [],
          },
        ],
    },
  });

  const query = new URLSearchParams(props.location.search); 

  if (window.Chart) {
    parseOptions(Chart, chartOptions());
  }

  useEffect(() => {
    fetch("http://127.0.0.1:4200/restaurant?id=" + query.get('id'), {
      method: 'get',
      headers: {
        'Accept': 'application/json, text/plain, */*',
        'Content-Type': 'application/json'
      },
    })
    .then(data => data.json())
    .then(res =>{

      //console.log(res);
      if(res["dining or delivery"] === "Dining and Delivery"){
          let diningKeys = [];
          let diningVals = [];
          let delKeys = [];
          let delVals = [];

          res.performance_dining.map((perf) => {
            diningKeys.push(...Object.keys(perf))
            diningVals.push(...Object.values(perf))
          })

          res.performance_delivery.map((perf) => {
            delKeys.push(...Object.keys(perf))
            delVals.push(...Object.values(perf))
          })

          chartExample1.current.data1.labels = diningKeys;
          chartExample1.current.data1.datasets[0].data = diningVals;
          chartExample1.current.data2.labels = delKeys;
          chartExample1.current.data2.datasets[0].data = delVals;

          //console.log(chartExample1.current)


      }
      else if(res["dining or delivery"] === "Dining"){
        let diningKeys = [];
        let diningVals = [];

        res.performance_dining.map((perf) => {
          diningKeys.push(...Object.keys(perf))
          diningVals.push(...Object.values(perf))
        })


        chartExample1.current.data1.labels = diningKeys;
        chartExample1.current.data1.datasets[0].data = diningVals;

      }
      else{
        let delKeys = [];
        let delVals = [];

        res.performance_delivery.map((perf) => {
          delKeys.push(...Object.keys(perf))
          delVals.push(...Object.values(perf))
        })

        chartExample1.current.data2.labels = delKeys;
        chartExample1.current.data2.datasets[0].data = delVals;

      }

      setRest(res);

      let myLatlng = new mapboxgl.LngLat(res.lng, res.lat);
      new mapboxgl.Marker({ "color": "orange"})
      .setLngLat(myLatlng)
      .addTo(map.current);
    });
  }, [activeNav])

  useEffect(() => {
    // returned function will be called on component unmount 
    return () => {
      Chart = initOptions;
    }
  }, [])

  const toggleNavs = (e, index) => {
    e.preventDefault();
    //console.log(chartExample1.current)
    setActiveNav(index);
    //console.log(index);
    setChartExample1Data("data" + index);
  };
  return (
    <>
      <Header1 rest={rest}/>
      {/* Page content */}
      <Container className="mt--7" fluid>
        <Row className="align-items-center">
          <Col className="mb-5 mb-xl-0" xl="8">
            <Card className="bg-gradient-default shadow">
              <CardHeader className="bg-transparent">
                <Row className="align-items-center">
                  <div className="col">
                    <h6 className="text-uppercase text-light ls-1 mb-1">
                      Overview
                    </h6>
                    <h2 className="text-white mb-0">Performance</h2>
                  </div>
                  <div className="col">
                    <Nav className="justify-content-end" pills>
                      <NavItem>
                        <NavLink
                          className={classnames("py-2 px-3", {
                            active: activeNav === 1,
                          })}
                          href=""
                          onClick={(e) => toggleNavs(e, 1)}
                        >
                          <span className="d-none d-md-block">Dining</span>
                          <span className="d-md-none">Di</span>
                        </NavLink>
                      </NavItem>
                      <NavItem>
                        <NavLink
                          className={classnames("py-2 px-3", {
                            active: activeNav === 2,
                          })}
                          data-toggle="tab"
                          href=""
                          onClick={(e) => toggleNavs(e, 2)}
                        >
                          <span className="d-none d-md-block">Delivery</span>
                          <span className="d-md-none">De</span>
                        </NavLink>
                      </NavItem>
                    </Nav>
                  </div>
                </Row>
              </CardHeader>
              <CardBody>
                {/* Chart */}
                <div className="chart">
                  <Line
                    data={chartExample1.current[chartExample1Data]}
                    options={chartExample1.current.options}
                    getDatasetAtEvent={(e) => console.log(e)}
                  />
                </div>
              </CardBody>
            </Card>
          </Col>
          <Col xl="4">
            <Card className="shadow">
              <CardHeader className="border-0">
                <Row className="align-items-center">
                  <div className="col">
                    <h3 className="mb-0">Location</h3>
                  </div>
                </Row>
              </CardHeader>
              <MapWrapper map={map}/>
            </Card>
          </Col>
          {/* <Col xl="4">
            <Card className="shadow">
              <CardHeader className="bg-transparent">
                <Row className="align-items-center">
                  <div className="col">
                    <h6 className="text-uppercase text-muted ls-1 mb-1">
                      Performance
                    </h6>
                    <h2 className="mb-0">Total orders</h2>
                  </div>
                </Row>
              </CardHeader>
              <CardBody>
                <div className="chart">
                  <Bar
                    data={chartExample2.data}
                    options={chartExample2.options}
                  />
                </div>
              </CardBody>
            </Card>
          </Col> */}
        </Row>
      </Container>
    </>
  );
};

export default Index;

import React, { useState, useEffect, useRef } from 'react';
import { useHistory } from "react-router-dom";
// reactstrap components
import {
  Badge,
  Card,
  CardHeader,
  CardFooter,
  Media,
  Pagination,
  PaginationItem,
  PaginationLink,
  Table,
  Container,
  Row,
} from "reactstrap";
// core components
import Header from "components/Headers/Header.js";

const Tables = () => {
  const [dining, setDining] = useState([]);
  const [delivery, setDelivery] = useState([]);

  const currDiningPage = useRef('1');
  const currDeliveryPage = useRef('1');

  useEffect(()=>{
    fetch("http://127.0.0.1:4200/leaderboard_dining?page=1", {
      method: 'get',
      headers: {
        'Accept': 'application/json, text/plain, */*',
        'Content-Type': 'application/json'
      },
    })
    .then(data => data.json())
    .then(res => setDining(res));

    fetch("http://127.0.0.1:4200/leaderboard_delivery?page=1", {
      method: 'get',
      headers: {
        'Accept': 'application/json, text/plain, */*',
        'Content-Type': 'application/json'
      },
    })
    .then(data => data.json())
    .then(res => setDelivery(res));

  }, []);

  const handleNextPage = ( type, page) =>{

      if(type==='dining'){
        fetch("http://127.0.0.1:4200/leaderboard_dining?page=" + page, {
          method: 'get',
          headers: {
            'Accept': 'application/json, text/plain, */*',
            'Content-Type': 'application/json'
          },
        })
        .then(data => data.json())
        .then(res =>{
          currDiningPage.current = page;
          setDining(res);
        });
      }
      else{
        fetch("http://127.0.0.1:4200/leaderboard_delivery?page=" + page, {
          method: 'get',
          headers: {
            'Accept': 'application/json, text/plain, */*',
            'Content-Type': 'application/json'
          },
        })
        .then(data => data.json())
        .then(res =>{
          currDeliveryPage.current = page;
          setDelivery(res);
        });
      }
  }

  const history = useHistory();
  const handleRowClick = (id) => {
    history.push(`/customer/restaurant/?id=${id}`);
  }  

  return (
    <>
      <Header />
      {/* Page content */}
      <Container className="mt--7" fluid>
        {/* Table */}
        <Row>
          <div className="col">
            <Card className="shadow">
              <CardHeader className="border-0">
                <h3 className="mb-0">Dining</h3>
              </CardHeader>
              <Table className="align-items-center table-flush" responsive>
                <thead className="thead-light">
                  <tr>
                    <th scope="col">Restaurant name</th>
                    <th scope="col">Cost for 2</th>
                    <th scope="col">Cuisines</th>
                    <th scope="col">Location</th>
                    <th scope="col">Rating</th>
                  </tr>
                </thead>
                <tbody>
                {dining.map((dine) => (
                  <tr key={dine.rest_id} style={{ cursor: "pointer" }} onClick={()=> handleRowClick(dine.rest_id)}>
                    <th scope="row">
                      <Media className="align-items-center">
                        <Media>
                          <span className="mb-0 text-sm">
                            {dine.name}
                          </span>
                        </Media>
                      </Media>
                    </th>
                    <td> &#x20b9;{dine.cost} </td>
                    <td>
                      <Badge color="" className="badge-dot mr-4">
                       {dine.cuisine}
                      </Badge>
                    </td>
                    <td>
                      {dine.locality_name}
                    </td>
                    <td>
                      {dine.dining_rating}
                    </td>
                  </tr>
                  ))}
                </tbody>
              </Table>
              <CardFooter className="py-4">
                <nav aria-label="...">
                  <Pagination
                    className="pagination justify-content-end mb-0"
                    listClassName="justify-content-end mb-0"
                  >
                    <PaginationItem active={currDiningPage.current==='1'}>
                      <PaginationLink
                        href="#pablo"
                        onClick={(e) => {
                          e.preventDefault();
                          handleNextPage('dining', '1')
                        } }
                      >
                        1
                      </PaginationLink>
                    </PaginationItem>
                    <PaginationItem active={currDiningPage.current==='2'}>
                      <PaginationLink
                        href="#pablo"
                        onClick={(e) => {
                          e.preventDefault();
                          handleNextPage('dining', '2')
                        } }
                      >
                        2
                      </PaginationLink>
                    </PaginationItem>
                    <PaginationItem active={currDiningPage.current==='3'}>
                      <PaginationLink
                        href="#pablo"
                        onClick={(e) => {
                          e.preventDefault();
                          handleNextPage('dining', '3')
                        } }
                      >
                        3
                      </PaginationLink>
                    </PaginationItem>
                    <PaginationItem active={currDiningPage.current==='4'}>
                      <PaginationLink
                        href="#pablo"
                        onClick={(e) => {
                          e.preventDefault();
                          handleNextPage('dining', '4')
                        } }
                      >
                        4
                      </PaginationLink>
                    </PaginationItem>
                    <PaginationItem active={currDiningPage.current==='5'}>
                      <PaginationLink
                        href="#pablo"
                        onClick={(e) => {
                          e.preventDefault();
                          handleNextPage('dining', '5')
                        } }
                      >
                        5
                      </PaginationLink>
                    </PaginationItem>
                  </Pagination>
                </nav>
              </CardFooter>
            </Card>
          </div>
        </Row>
        {/* Dark table */}
        <Row className="mt-5">
          <div className="col">
            <Card className="bg-default shadow">
              <CardHeader className="bg-transparent border-0">
                <h3 className="text-white mb-0">Delivery</h3>
              </CardHeader>
              <Table
                className="align-items-center table-dark table-flush"
                responsive
              >
                <thead className="thead-dark">
                  <tr>
                    <th scope="col">Restaurant name</th>
                    <th scope="col">Cost for 2</th>
                    <th scope="col">Cuisnes</th>
                    <th scope="col">Location</th>
                    <th scope="col">Rating</th>
                  </tr>
                </thead>
                <tbody>
                  {delivery.map((del) => (
                  <tr key={del.rest_id} style={{ cursor: "pointer" }} onClick={()=> handleRowClick(del.rest_id)}>
                    <th scope="row">
                      <Media className="align-items-center">
                        <Media>
                          <span className="mb-0 text-sm">
                            {del.name}
                          </span>
                        </Media>
                      </Media>
                    </th>
                    <td> &#x20b9;{del.cost} </td>
                    <td>
                      <Badge color="" className="badge-dot mr-4">
                       {del.cuisine}
                      </Badge>
                    </td>
                    <td>
                      {del.locality_name}
                    </td>
                    <td>
                      {del.delivery_rating}
                    </td>
                  </tr>
                  ))}
                </tbody>
              </Table>
              <CardFooter className="py-4 bg-default shadow">
                <nav aria-label="...">
                  <Pagination
                    className="pagination justify-content-end mb-0"
                    listClassName="justify-content-end mb-0"
                  >
                    <PaginationItem active={currDeliveryPage.current==='1'}>
                      <PaginationLink
                        href="#pablo"
                        onClick={(e) => {
                          e.preventDefault();
                          handleNextPage('delivery', '1')
                        } }
                      >
                        1
                      </PaginationLink>
                    </PaginationItem>
                    <PaginationItem active={currDeliveryPage.current==='2'}>
                      <PaginationLink
                        href="#pablo"
                        onClick={(e) => {
                          e.preventDefault();
                          handleNextPage('delivery', '2')
                        } }
                      >
                        2
                      </PaginationLink>
                    </PaginationItem>
                    <PaginationItem active={currDeliveryPage.current==='3'}>
                      <PaginationLink
                        href="#pablo"
                        onClick={(e) => {
                          e.preventDefault();
                          handleNextPage('delivery', '3')
                        } }
                      >
                        3
                      </PaginationLink>
                    </PaginationItem>
                    <PaginationItem active={currDeliveryPage.current==='4'}>
                      <PaginationLink
                        href="#pablo"
                        onClick={(e) => {
                          e.preventDefault();
                          handleNextPage('delivery', '4')
                        } }
                      >
                        4
                      </PaginationLink>
                    </PaginationItem>
                    <PaginationItem active={currDeliveryPage.current==='5'}>
                      <PaginationLink
                        href="#pablo"
                        onClick={(e) => {
                          e.preventDefault();
                          handleNextPage('delivery', '5')
                        } }
                      >
                        5
                      </PaginationLink>
                    </PaginationItem>
                  </Pagination>
                </nav>
              </CardFooter>
            </Card>
          </div>
        </Row>
      </Container>
    </>
  );
};

export default Tables;

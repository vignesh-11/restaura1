import React, { useState } from 'react';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableContainer from '@material-ui/core/TableContainer';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import Paper from '@material-ui/core/Paper';
import { CssBaseline, Container, Box, Button} from "@material-ui/core";


export default function Reports() {

  const [rows, setRows] = React.useState(JSON.parse(localStorage.getItem("reports") || "[]"));

  const showDetails = (data) => {
        let text = "";
        text += "name : " + data.name + "\n" ;
        text += "online order : " + data.online_order + "\n";
        text += "online book table : " + data.book_table + "\n";
        text += "type : " + data.rest_type + "\n";
        text += "cuisines : " + data.cuisines + "\n";
        text += "cost : " + data["approx_cost(for two people)"] + "\n";
        text += "location : " + data.location + "\n";
        alert(text);
  }

  const handleDownload = (data) => {
    fetch("http://127.0.0.1:5000/report", {
      method: 'post',
      headers: {
        'Accept': 'application/json, text/plain, */*',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    })
    .then(response => response.blob())
    .then(blob => URL.createObjectURL(blob))
    .then(url => {
        window.open(url, '_blank');
        URL.revokeObjectURL(url);
    });
  }

  const handleDelete = (id) => {
    var reports = JSON.parse(localStorage.getItem("reports") || "[]");
    reports = reports.filter(item => item.id !== id)
    //console.log(reports);
    localStorage.setItem("reports", JSON.stringify(reports));
    setRows(rows.filter(item => item.id !== id));
  }

  return (
      <>
        <CssBaseline />
        <Container component={Box} p={4}>
        <Paper component={Box} p={3}>
            <TableContainer component={Paper}>
            <Table sx={{ minWidth: 650 }} aria-label="simple table">
                <TableHead>
                <TableRow>
                    <TableCell><b>Name</b></TableCell>
                    <TableCell align="right">&nbsp;</TableCell>
                    <TableCell align="right">&nbsp;</TableCell>
                    <TableCell align="right">&nbsp;</TableCell>
                </TableRow>
                </TableHead>
                <TableBody>
                {rows.map((row) => (
                    <TableRow
                    key={row.id}
                    sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                    >
                    <TableCell component="th" scope="row">
                        {row.name}
                    </TableCell>
                    <TableCell align="right"><Button onClick={() => showDetails(row)} variant="contained">show details</Button></TableCell>
                    <TableCell align="right"><Button variant="contained" onClick={() => handleDownload(row)}>Download</Button></TableCell>
                    <TableCell align="right"><Button onClick={() => handleDelete(row.id)} variant="outlined">Delete</Button></TableCell>
                    </TableRow>
                ))}
                </TableBody>
            </Table>
            </TableContainer>
        </Paper>
        </Container>
    </>
  );
}

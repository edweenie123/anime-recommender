<script setup>
import { RouterLink, RouterView } from 'vue-router'
import Header from './components/Header.vue'
import SearchBar from './components/SearchBar.vue'
import AnimeCards from './components/AnimeCards.vue'
// import HelloWorld from '@/components/HelloWorld.vue'
</script>

<template>
  <div class="container" id="app">
    <Header />
    <SearchBar />
    <AnimeCards />
    <!-- <RouterView /> -->
  </div>
  <!-- <header> -->
  <!--   <img alt="Vue logo" class="logo" src="@/assets/logo.svg" width="125" height="125" /> -->

  <!--   <div class="wrapper"> -->
  <!--     <HelloWorld msg="You did it!" /> -->

  <!--     <nav> -->
  <!--       <RouterLink to="/">Home</RouterLink> -->
  <!--       <RouterLink to="/about">About</RouterLink> -->
  <!--     </nav> -->
  <!--   </div> -->
  <!-- </header> -->

</template>

<style>
/* @import url("https://fonts.googleapis.com/css2?family=Montserrat&display=swap"); */
/* * { */
/*   padding: 0; */
/*   margin: 0; */
/*   box-sizing: border-box; */
/*   font-family: "Montserrat", sans-serif; */
/* } */

/* body { */
/*   padding: 20px; */
/*   min-height: 100vh; */
/*   background-color: rgb(234, 242, 255); */
/* } */



</style>

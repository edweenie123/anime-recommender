<script setup>
import { RouterLink, RouterView } from "vue-router";
import Header from "./components/Header.vue";
import SearchBar from "./components/SearchBar.vue";
import AnimeCards from "./components/AnimeCards.vue";
import Navbar from "./components/Navbar.vue";
// import HelloWorld from '@/components/HelloWorld.vue'
</script>

<template>
  <div id="app">
    <Navbar />
    <div class="container" id="app">
      <!-- <Header /> -->
      <!-- <SearchBar @upd-anime-list="updateAnimeList" /> -->
      <!-- <AnimeCards :animeList="animeList" /> -->
      <RouterView />
    </div>
  </div>
  <!-- <header> -->
  <!--   <img alt="Vue logo" class="logo" src="@/assets/logo.svg" width="125" height="125" /> -->

  <!-- <div class="wrapper"> -->
  <!--   <HelloWorld msg="You did it!" /> -->

  <!--   <nav> -->
  <!--     <RouterLink to="/">Home</RouterLink> -->
  <!--     <RouterLink to="/about">About</RouterLink> -->
  <!--   </nav> -->
  <!-- </div> -->
  <!-- </header> -->
</template>

<script>
export default {
  data() {
    return {
      animeList: [],
    };
  },
  methods: {
    updateAnimeList(newList) {
      this.animeList = newList;
    },
  },
};
</script>

<style>
/* #app { */
/*   margin-top: 60px; */
/* } */
#bruh {
  border: 1px solid blue;
}

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
